from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
def _as(self, methname, builder, value):
    extracted = []
    for i, dm in enumerate(self._models):
        extracted.append(getattr(dm, methname)(builder, self.get(builder, value, i)))
    return tuple(extracted)