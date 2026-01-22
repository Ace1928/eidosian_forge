import collections
import logging
import math
import random
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from transformers import UdopConfig
from transformers.modeling_outputs import (
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from ..deprecated._archive_maps import UDOP_PRETRAINED_MODEL_ARCHIVE_LIST  # noqa: F401, E402
class RelativePositionBiasAggregated(nn.Module):

    def __init__(self, modules: Sequence[RelativePositionBiasBase]):
        """
        Class which sums up various computed biases.

        Args:
            modules (Sequence[RelativePositionBiasBase]):
                List of relative bias modules.
        """
        super().__init__()
        self.biases = nn.ModuleList(modules)

    def forward(self, attention_mask: Optional[Tensor]=None, bbox: Optional[Dict[str, Any]]=None) -> Union[float, Tensor]:
        output = 0.0
        for bias in self.biases:
            output = bias(attention_mask, bbox) + output
        return output