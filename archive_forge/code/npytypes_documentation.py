import collections
import warnings
from functools import cached_property
from llvmlite import ir
from .abstract import DTypeSpec, IteratorType, MutableSequence, Number, Type
from .common import Buffer, Opaque, SimpleIteratorType
from numba.core.typeconv import Conversion
from numba.core import utils
from .misc import UnicodeType
from .containers import Bytes
import numpy as np

        Convert this type to the corresponding pointer type.
        This allows passing a array.ctypes object to a C function taking
        a raw pointer.

        Note that in pure Python, the array.ctypes object can only be
        passed to a ctypes function accepting a c_void_p, not a typed
        pointer.
        