import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_realm import RealmConfig
@dataclass
class RealmForOpenQAOutput(ModelOutput):
    """

    Outputs of [`RealmForOpenQA`] models.

    Args:
        reader_output (`dict`):
            Reader output.
        predicted_answer_ids (`torch.LongTensor` of shape `(answer_sequence_length)`):
            Predicted answer ids.
    """
    reader_output: dict = None
    predicted_answer_ids: torch.LongTensor = None