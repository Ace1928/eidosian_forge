import collections
from contextlib import contextmanager, ExitStack
import functools
from llvmlite import ir
from numba.core import utils, types, config, debuginfo
import numba.core.datamodel
def _setvalue(self, value):
    """Store the value in this structure"""
    assert not is_pointer(value.type)
    assert value.type == self._type, (value.type, self._type)
    self._builder.store(value, self._value)