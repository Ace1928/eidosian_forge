from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
@register_default(types.Omitted)
class OmittedArgDataModel(DataModel):
    """
    A data model for omitted arguments.  Only the "argument" representation
    is defined, other representations raise a NotImplementedError.
    """

    def get_value_type(self):
        return ir.LiteralStructType([])

    def get_argument_type(self):
        return ()

    def as_argument(self, builder, val):
        return ()

    def from_argument(self, builder, val):
        assert val == (), val
        return None