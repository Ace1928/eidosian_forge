import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_align import AlignConfig, AlignTextConfig, AlignVisionConfig
def to_tuple(self) -> Tuple[Any]:
    return tuple((self[k] if k not in ['text_model_output', 'vision_model_output'] else getattr(self, k).to_tuple() for k in self.keys()))