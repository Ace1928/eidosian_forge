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
def tuple_new(self, count):
    fnty = ir.FunctionType(self.pyobj, [ir.IntType(32)])
    fn = self._get_function(fnty, name='PyTuple_New')
    return self.builder.call(fn, [self.context.get_constant(types.int32, count)])