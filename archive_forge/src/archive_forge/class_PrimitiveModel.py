from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
class PrimitiveModel(DataModel):
    """A primitive type can be represented natively in the target in all
    usage contexts.
    """

    def __init__(self, dmm, fe_type, be_type):
        super(PrimitiveModel, self).__init__(dmm, fe_type)
        self.be_type = be_type

    def get_value_type(self):
        return self.be_type

    def as_data(self, builder, value):
        return value

    def as_argument(self, builder, value):
        return value

    def as_return(self, builder, value):
        return value

    def from_data(self, builder, value):
        return value

    def from_argument(self, builder, value):
        return value

    def from_return(self, builder, value):
        return value