from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
class CompositeModel(DataModel):
    """Any model that is composed of multiple other models should subclass from
    this.
    """
    pass