from abc import ABCMeta, abstractmethod
from dataclasses import asdict, dataclass
from typing import Optional, Type, TypeVar
import torch
import torch.nn as nn
from xformers.components.attention import AttentionMask
@staticmethod
def _maybe_pad_sequence(x: torch.Tensor, mask: torch.Tensor):
    """
        If the sequence is shorter than the mask, return a padded view
        """
    if x.shape[-2] != mask.shape[-1]:
        assert x.shape[-2] < mask.shape[-1], 'Sequence is bigger than the provided mask, cannot infer what to do with it. Please update your attention mask'
        pad_size = (0, 0, 0, mask.shape[-1] - x.shape[-2], 0, 0)
        return torch.nn.functional.pad(x, pad_size, mode='constant', value=0.0)
    return x