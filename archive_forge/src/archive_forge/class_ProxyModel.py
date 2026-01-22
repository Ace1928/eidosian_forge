from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
class ProxyModel(DataModel):
    """
    Helper class for models which delegate to another model.
    """

    def get_value_type(self):
        return self._proxied_model.get_value_type()

    def get_data_type(self):
        return self._proxied_model.get_data_type()

    def get_return_type(self):
        return self._proxied_model.get_return_type()

    def get_argument_type(self):
        return self._proxied_model.get_argument_type()

    def as_data(self, builder, value):
        return self._proxied_model.as_data(builder, value)

    def as_argument(self, builder, value):
        return self._proxied_model.as_argument(builder, value)

    def as_return(self, builder, value):
        return self._proxied_model.as_return(builder, value)

    def from_data(self, builder, value):
        return self._proxied_model.from_data(builder, value)

    def from_argument(self, builder, value):
        return self._proxied_model.from_argument(builder, value)

    def from_return(self, builder, value):
        return self._proxied_model.from_return(builder, value)