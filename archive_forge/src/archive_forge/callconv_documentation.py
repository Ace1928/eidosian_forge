from collections import namedtuple
from collections.abc import Iterable
import itertools
import hashlib
from llvmlite import ir
from numba.core import types, cgutils, errors
from numba.core.base import PYOBJECT, GENERIC_POINTER

        Call the Numba-compiled *callee*.
        Parameters:
        -----------
        attrs: LLVM style string or iterable of individual attributes, default
               is None which specifies no attributes. Examples:
               LLVM style string: "noinline fast"
               Equivalent iterable: ("noinline", "fast")
        