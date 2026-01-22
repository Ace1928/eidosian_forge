import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, ModelOutput, QuestionAnsweringModelOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_splinter import SplinterConfig
def _prepare_question_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
    rows, flat_positions = torch.where(input_ids == self.config.question_token_id)
    num_questions = torch.bincount(rows)
    positions = torch.full((input_ids.size(0), num_questions.max()), self.config.pad_token_id, dtype=torch.long, device=input_ids.device)
    cols = torch.cat([torch.arange(n) for n in num_questions])
    positions[rows, cols] = flat_positions
    return positions