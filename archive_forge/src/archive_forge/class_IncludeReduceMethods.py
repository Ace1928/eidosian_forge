from __future__ import annotations
import operator
import numpy as np
from xarray.core import dtypes, duck_array_ops
class IncludeReduceMethods:
    __slots__ = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if getattr(cls, '_reduce_method', None):
            inject_reduce_methods(cls)