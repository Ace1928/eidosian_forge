import abc
import torch
import itertools
import collections
from torch.nn.modules.module import _addindent
def _ntuple_from_first(n):
    """Converts the argument to a tuple of size n
    with the first element repeated."""

    def parse(x):
        while isinstance(x, collections.abc.Sequence):
            if len(x) == n:
                break
            x = x[0]
        return tuple(itertools.repeat(x, n))
    return parse