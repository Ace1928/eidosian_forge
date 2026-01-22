import typing
from typing import Any, Callable, List, Union, cast, Sequence
import torch
from torch import Tensor
import torch.cuda.comm
def _setitem_by_index(self, index: int, value) -> None:
    if not self.atomic:
        i = index
        self._values = self._values[:i] + (value,) + self._values[i + 1:]
        return
    if index != 0:
        raise IndexError('atomic batch allows index 0 only')
    self._values = value