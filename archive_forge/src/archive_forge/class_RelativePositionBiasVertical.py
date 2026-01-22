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
class RelativePositionBiasVertical(RelativePositionBiasBase):

    def __init__(self, scaling_factor=100, max_distance=100, **kwargs):
        """
        Represents in the bucket embeddings vertical distance between two tokens. Parameters are the same as in base
        class
        """
        super().__init__(scaling_factor=scaling_factor, max_distance=max_distance, **kwargs)

    def prepare_input(self, attention_mask: Optional[Tensor]=None, bbox: Optional[Dict[str, Any]]=None) -> Tensor:
        if not self.scaling_factor > 1.0:
            raise ValueError('Need to scale the values of bboxes, as there are in small (0,1) range')
        if bbox is None:
            raise ValueError('Bbox is required for vertical relative position bias')
        vertical_position: Tensor = bbox[:, :, [1, 3]].mean(dim=-1)
        return self.get_relative_position(vertical_position)