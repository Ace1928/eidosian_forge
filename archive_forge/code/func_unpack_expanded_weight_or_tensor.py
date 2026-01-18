from typing import Optional
import torch
from .expanded_weights_impl import ExpandedWeight
def unpack_expanded_weight_or_tensor(maybe_expanded_weight, func=lambda x: x):
    if isinstance(maybe_expanded_weight, ExpandedWeight):
        orig_weight = maybe_expanded_weight.orig_weight
        return func(orig_weight)
    elif isinstance(maybe_expanded_weight, torch.Tensor) and (not maybe_expanded_weight.requires_grad):
        return func(maybe_expanded_weight)
    elif isinstance(maybe_expanded_weight, torch.Tensor):
        raise RuntimeError('ExpandedWeights currently does not support a mixture of ExpandedWeight parameters and normal Parameters. Please file and issue with pytorch/pytorch')