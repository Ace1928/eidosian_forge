from .module import Module
from typing import Tuple, Union
from torch import Tensor
from torch.types import _size
def _require_tuple_tuple(self, input):
    if isinstance(input, tuple):
        for idx, elem in enumerate(input):
            if not isinstance(elem, tuple):
                raise TypeError('unflattened_size must be tuple of tuples, ' + f'but found element of type {type(elem).__name__} at pos {idx}')
        return
    raise TypeError('unflattened_size must be a tuple of tuples, ' + f'but found type {type(input).__name__}')