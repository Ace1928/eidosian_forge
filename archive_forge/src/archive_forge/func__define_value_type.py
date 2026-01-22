from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
def _define_value_type(self, value_type):
    if value_type.is_opaque:
        value_type.set_body(self._actual_model.get_value_type())