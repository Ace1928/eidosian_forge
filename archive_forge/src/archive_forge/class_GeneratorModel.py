from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
@register_default(types.Generator)
class GeneratorModel(CompositeModel):

    def __init__(self, dmm, fe_type):
        super(GeneratorModel, self).__init__(dmm, fe_type)
        self._arg_models = [self._dmm.lookup(t) for t in fe_type.arg_types if not isinstance(t, types.Omitted)]
        self._state_models = [self._dmm.lookup(t) for t in fe_type.state_types]
        self._args_be_type = ir.LiteralStructType([t.get_data_type() for t in self._arg_models])
        self._state_be_type = ir.LiteralStructType([t.get_data_type() for t in self._state_models])
        self._be_type = ir.LiteralStructType([self._dmm.lookup(types.int32).get_value_type(), self._args_be_type, self._state_be_type])
        self._be_ptr_type = self._be_type.as_pointer()

    def get_value_type(self):
        """
        The generator closure is passed around as a reference.
        """
        return self._be_ptr_type

    def get_argument_type(self):
        return self._be_ptr_type

    def get_return_type(self):
        return self._be_type

    def get_data_type(self):
        return self._be_type

    def as_argument(self, builder, value):
        return value

    def from_argument(self, builder, value):
        return value

    def as_return(self, builder, value):
        return self.as_data(builder, value)

    def from_return(self, builder, value):
        return self.from_data(builder, value)

    def as_data(self, builder, value):
        return builder.load(value)

    def from_data(self, builder, value):
        stack = cgutils.alloca_once(builder, value.type)
        builder.store(value, stack)
        return stack