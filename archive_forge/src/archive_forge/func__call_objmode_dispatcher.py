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
@staticmethod
def _call_objmode_dispatcher(compile_args):
    dispatcher, argtypes = compile_args
    entrypt = dispatcher.compile(argtypes)
    return entrypt