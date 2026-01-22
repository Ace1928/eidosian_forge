import enum
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, MaskedLMOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import (
from ...utils import (
from .configuration_tapas import TapasConfig
def _calculate_regression_loss(answer, aggregate_mask, dist_per_cell, numeric_values, numeric_values_scale, input_mask_float, logits_aggregation, config):
    """
    Calculates the regression loss per example.

    Args:
        answer (`torch.FloatTensor` of shape `(batch_size,)`):
            Answer for every example in the batch. Nan if there is no scalar answer.
        aggregate_mask (`torch.FloatTensor` of shape `(batch_size,)`):
            A mask set to 1 for examples that should use aggregation functions.
        dist_per_cell (`torch.distributions.Bernoulli`):
            Cell selection distribution for each cell.
        numeric_values (`torch.FloatTensor` of shape `(batch_size, seq_length)`):
            Numeric values of every token. Nan for tokens which are not numeric values.
        numeric_values_scale (`torch.FloatTensor` of shape `(batch_size, seq_length)`):
            Scale of the numeric values of every token.
        input_mask_float (`torch.FloatTensor` of shape `(batch_size, seq_length)`):
            Mask for the table, without question tokens and table headers.
        logits_aggregation (`torch.FloatTensor` of shape `(batch_size, num_aggregation_labels)`):
            Logits per aggregation operation.
        config ([`TapasConfig`]):
            Model configuration class with all the parameters of the model

    Returns:
        per_example_answer_loss_scaled (`torch.FloatTensor` of shape `(batch_size,)`): Scales answer loss for each
        example in the batch. large_answer_loss_mask (`torch.FloatTensor` of shape `(batch_size,)`): A mask which is 1
        for examples for which their answer loss is larger than the answer_loss_cutoff.
    """
    expected_result = _calculate_expected_result(dist_per_cell, numeric_values, numeric_values_scale, input_mask_float, logits_aggregation, config)
    answer_masked = torch.where(torch.isnan(answer), torch.zeros_like(answer), answer)
    if config.use_normalized_answer_loss:
        normalizer = (torch.max(torch.abs(expected_result), torch.abs(answer_masked)) + EPSILON_ZERO_DIVISION).detach()
        normalized_answer_masked = answer_masked / normalizer
        normalized_expected_result = expected_result / normalizer
        per_example_answer_loss = huber_loss(normalized_expected_result * aggregate_mask, normalized_answer_masked * aggregate_mask)
    else:
        per_example_answer_loss = huber_loss(expected_result * aggregate_mask, answer_masked * aggregate_mask, delta=config.huber_loss_delta)
    if config.answer_loss_cutoff is None:
        large_answer_loss_mask = torch.ones_like(per_example_answer_loss, dtype=torch.float32)
    else:
        large_answer_loss_mask = torch.where(per_example_answer_loss > config.answer_loss_cutoff, torch.zeros_like(per_example_answer_loss, dtype=torch.float32), torch.ones_like(per_example_answer_loss, dtype=torch.float32))
    per_example_answer_loss_scaled = config.answer_loss_importance * (per_example_answer_loss * aggregate_mask)
    return (per_example_answer_loss_scaled, large_answer_loss_mask)