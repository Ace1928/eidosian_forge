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
def _segment_reduce(values, index, segment_reduce_fn, name):
    """
    Applies a segment reduction segment-wise.

    Args:
        values (`torch.Tensor`):
            Tensor with segment values.
        index (`IndexMap`):
            IndexMap.
        segment_reduce_fn (`str`):
            Name for the reduce operation. One of "sum", "mean", "max" or "min".
        name (`str`):
            Name for the operation. Currently not used

    Returns:
        (`IndexMap`): IndexMap of shape batch_shape with elements equal to range(num_segments).
    """
    flat_index = flatten(index)
    vector_shape = values.size()[len(index.indices.size()):]
    flattened_shape = torch.cat([torch.as_tensor([-1], dtype=torch.long), torch.as_tensor(vector_shape, dtype=torch.long)], dim=0)
    flat_values = values.reshape(flattened_shape.tolist())
    out = torch.zeros(int(flat_index.num_segments), dtype=torch.float, device=flat_values.device)
    segment_means = out.scatter_reduce(dim=0, index=flat_index.indices.long(), src=flat_values.float(), reduce=segment_reduce_fn, include_self=False)
    new_shape = torch.cat([torch.as_tensor(index.batch_shape(), dtype=torch.long), torch.as_tensor([index.num_segments], dtype=torch.long), torch.as_tensor(vector_shape, dtype=torch.long)], dim=0)
    output_values = segment_means.clone().view(new_shape.tolist()).to(values.dtype)
    output_index = range_index_map(index.batch_shape(), index.num_segments)
    return (output_values, output_index)