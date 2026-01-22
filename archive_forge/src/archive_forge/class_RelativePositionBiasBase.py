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
class RelativePositionBiasBase(nn.Module, ABC):
    """
    Base class of relative biases.

    Args:
        num_heads (`int`):
            Number of attention heads in the model, it will create embeddings of size `num_heads`, which will be added to the scores of each token pair.
        relative_attention_num_buckets (`int`, *optional*, defaults to 32):
            Pair token metric (distance in the sequence, distance in pixels etc.) will be bucketed, parameter is defining number of such
            buckets.
        bidirectional (`bool`, *optional*, defaults to `True`):
            Whether the distance should be bidirectional for a pair of tokens. If `False`, then distance(tok1, tok2) == distance(tok2, tok1).
        scaling_factor (`int`, *optional*, defaults to 1):
            Defining factor which will be used to scale relative distance.
        max_distance (`int`, *optional*, defaults to 128):
            All distances above this value will end up in the one/same bucket.
        augmentation (`bool`, *optional*, defaults to `False`):
            Whether to multiply relative distances by a random scalar.
        expand (`bool`, *optional*, defaults to `False`):
            Whether to expand an existing pretrained model with subsequent additions of prefix_bucket.
    """

    def __init__(self, num_heads=None, relative_attention_num_buckets=32, bidirectional=True, scaling_factor=1, max_distance=128, level='tokens', augmentation=False, prefix_bucket=False, expand=False):
        super(RelativePositionBiasBase, self).__init__()
        self.prefix_bucket = prefix_bucket
        self.augmentation = augmentation
        self.level = level
        self.max_distance = max_distance
        self.scaling_factor = scaling_factor
        self.bidirectional = bidirectional
        self.num_heads = num_heads
        self.expand = expand
        self.relative_attention_num_buckets = relative_attention_num_buckets
        extra_head = 2 if prefix_bucket and (not self.expand) else 0
        self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets + extra_head, self.num_heads)

    @abstractmethod
    def prepare_input(self, attention_mask: Optional[Tensor]=None, bbox: Optional[Dict[str, Any]]=None) -> Tensor:
        pass

    def get_bucket(self, attention_mask: Optional[Tensor]=None, bbox: Optional[Dict[str, Any]]=None) -> Tensor:
        relative_position = self.prepare_input(attention_mask, bbox)
        rp_bucket: Tensor = get_relative_position_bucket(relative_position, bidirectional=self.bidirectional, num_buckets=self.relative_attention_num_buckets, max_distance=self.max_distance)
        return rp_bucket

    def get_relative_position(self, positions):
        context_position = positions[:, :, None]
        memory_position = positions[:, None, :]
        relative_position = memory_position - context_position
        if self.augmentation and self.training:
            relative_position *= random.uniform(*AUGMENTATION_RANGE)
        relative_position *= self.scaling_factor
        return relative_position.to(torch.long)

    def forward(self, attention_mask: Optional[Tensor]=None, bbox: Optional[Dict[str, Any]]=None) -> Tensor:
        if self.expand and self.prefix_bucket:
            new_bias = nn.Embedding(self.relative_attention_num_buckets + 2, self.num_heads)
            new_bias.weight.data[:self.relative_attention_num_buckets] = self.relative_attention_bias.weight.data
            new_bias.weight.data[self.relative_attention_num_buckets:] = 0.1
            self.relative_attention_bias = new_bias
            self.expand = False
        rp_bucket = self.get_bucket(attention_mask, bbox)
        if self.prefix_bucket:
            if rp_bucket.size(0) == 1 and attention_mask.size(0) > 1:
                rp_bucket = rp_bucket.repeat(attention_mask.size(0), 1, 1)
            is_prefix = bbox[:, :, 1] < 0
            num_prefix = is_prefix.sum(-1)
            for idx, num_prefix_row in enumerate(num_prefix.cpu().numpy()):
                rp_bucket[idx, :num_prefix_row, num_prefix_row:] = self.relative_attention_num_buckets
                rp_bucket[idx, num_prefix_row:, :num_prefix_row] = self.relative_attention_num_buckets + 1
        values: Tensor = self.relative_attention_bias(rp_bucket)
        if values.dim() != 4:
            raise ValueError('Wrong dimension of values tensor')
        values = values.permute([0, 3, 1, 2])
        return values