import collections
from contextlib import contextmanager, ExitStack
import functools
from llvmlite import ir
from numba.core import utils, types, config, debuginfo
import numba.core.datamodel
def _cast_member_from_value(self, index, val):
    model = self._datamodel.get_model(index)
    return model.as_data(self._builder, val)