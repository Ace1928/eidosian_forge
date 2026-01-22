import abc
import torch
import itertools
import collections
from torch.nn.modules.module import _addindent
class WeightedQuantizedModule(torch.nn.Module, metaclass=abc.ABCMeta):
    """Wrapper for quantized modules than can be lowered from reference modules."""

    @classmethod
    @abc.abstractmethod
    def from_reference(cls, ref_module, output_scale, output_zero_point):
        raise NotImplementedError