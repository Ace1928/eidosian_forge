import dis
import inspect
from typing import Sequence, Union
import torch
import functorch._C
from functorch._C import dim as _C
from .tree_map import tree_flatten, tree_map
from .wrap_type import wrap_type
from . import op_properties
class _Tensor:

    @property
    def dims(self):
        return tuple((d for d in self._levels if isinstance(d, Dim)))

    def dim(self):
        return self.ndim
    if use_c:
        __torch_function__ = classmethod(_C.__torch_function__)
        expand = _C._instancemethod(_C.expand)
    else:
        __torch_function__ = reference.__torch_function__
        expand = reference.expand
    index = _C._instancemethod(_C.index)

    def __repr__(self):
        tensor, levels, ndim = (self._tensor, self._levels, self.ndim)
        return f'{tensor}\nwith dims={tuple((l + ndim if isinstance(l, int) else l for l in levels))} sizes={tuple(tensor.size())}'