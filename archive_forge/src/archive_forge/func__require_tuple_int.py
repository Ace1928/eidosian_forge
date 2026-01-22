from .module import Module
from typing import Tuple, Union
from torch import Tensor
from torch.types import _size
def _require_tuple_int(self, input):
    if isinstance(input, (tuple, list)):
        for idx, elem in enumerate(input):
            if not isinstance(elem, int):
                raise TypeError('unflattened_size must be tuple of ints, ' + f'but found element of type {type(elem).__name__} at pos {idx}')
        return
    raise TypeError(f'unflattened_size must be a tuple of ints, but found type {type(input).__name__}')