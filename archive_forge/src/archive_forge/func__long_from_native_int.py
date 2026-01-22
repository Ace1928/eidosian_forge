from collections import namedtuple
import contextlib
import pickle
import hashlib
import sys
from llvmlite import ir
from llvmlite.ir import Constant
import ctypes
from numba import _helperlib
from numba.core import (
from numba.core.utils import PYVERSION
def _long_from_native_int(self, ival, func_name, native_int_type, signed):
    fnty = ir.FunctionType(self.pyobj, [native_int_type])
    fn = self._get_function(fnty, name=func_name)
    resptr = cgutils.alloca_once(self.builder, self.pyobj)
    fn = self._get_function(fnty, name=func_name)
    self.builder.store(self.builder.call(fn, [ival]), resptr)
    return self.builder.load(resptr)