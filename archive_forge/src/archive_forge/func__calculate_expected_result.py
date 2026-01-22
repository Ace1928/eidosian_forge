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
def _calculate_expected_result(dist_per_cell, numeric_values, numeric_values_scale, input_mask_float, logits_aggregation, config):
    """
    Calculates the expected result given cell and aggregation probabilities.

    Args:
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
            Model configuration class with all the hyperparameters of the model

    Returns:
        expected_result (`torch.FloatTensor` of shape `(batch_size,)`): The expected result per example.
    """
    if config.use_gumbel_for_cells:
        gumbel_dist = torch.distributions.RelaxedBernoulli(temperature=config.temperature, logits=dist_per_cell.logits * config.temperature)
        scaled_probability_per_cell = gumbel_dist.sample()
    else:
        scaled_probability_per_cell = dist_per_cell.probs
    scaled_probability_per_cell = scaled_probability_per_cell / numeric_values_scale * input_mask_float
    count_result = torch.sum(scaled_probability_per_cell, dim=1)
    numeric_values_masked = torch.where(torch.isnan(numeric_values), torch.zeros_like(numeric_values), numeric_values)
    sum_result = torch.sum(scaled_probability_per_cell * numeric_values_masked, dim=1)
    avg_approximation = config.average_approximation_function
    if avg_approximation == AverageApproximationFunction.RATIO:
        average_result = sum_result / (count_result + EPSILON_ZERO_DIVISION)
    elif avg_approximation == AverageApproximationFunction.FIRST_ORDER:
        ex = torch.sum(scaled_probability_per_cell, dim=1, keepdim=True) - scaled_probability_per_cell + 1
        average_result = torch.sum(numeric_values_masked * scaled_probability_per_cell / ex, dim=1)
    elif avg_approximation == AverageApproximationFunction.SECOND_ORDER:
        ex = torch.sum(scaled_probability_per_cell, dim=1, keepdim=True) - scaled_probability_per_cell + 1
        pointwise_var = scaled_probability_per_cell * (1 - scaled_probability_per_cell)
        var = torch.sum(pointwise_var, dim=1, keepdim=True) - pointwise_var
        multiplier = (var / torch.square(ex) + 1) / ex
        average_result = torch.sum(numeric_values_masked * scaled_probability_per_cell * multiplier, dim=1)
    else:
        raise ValueError(f'Invalid average_approximation_function: {config.average_approximation_function}')
    if config.use_gumbel_for_aggregation:
        gumbel_dist = torch.distributions.RelaxedOneHotCategorical(config.aggregation_temperature, logits=logits_aggregation[:, 1:])
        aggregation_op_only_probs = gumbel_dist.sample()
    else:
        aggregation_op_only_probs = nn.functional.softmax(logits_aggregation[:, 1:] / config.aggregation_temperature, dim=-1)
    all_results = torch.cat([torch.unsqueeze(sum_result, dim=1), torch.unsqueeze(average_result, dim=1), torch.unsqueeze(count_result, dim=1)], dim=1)
    expected_result = torch.sum(all_results * aggregation_op_only_probs, dim=1)
    return expected_result