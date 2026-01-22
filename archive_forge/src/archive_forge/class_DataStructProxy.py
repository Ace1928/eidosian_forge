import collections
from contextlib import contextmanager, ExitStack
import functools
from llvmlite import ir
from numba.core import utils, types, config, debuginfo
import numba.core.datamodel
class DataStructProxy(_StructProxy):
    """
    Create a StructProxy suitable for accessing data persisted in memory.
    """

    def _get_be_type(self, datamodel):
        return datamodel.get_data_type()

    def _cast_member_to_value(self, index, val):
        model = self._datamodel.get_model(index)
        return model.from_data(self._builder, val)

    def _cast_member_from_value(self, index, val):
        model = self._datamodel.get_model(index)
        return model.as_data(self._builder, val)