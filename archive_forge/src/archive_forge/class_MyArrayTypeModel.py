import builtins
import unittest
from numbers import Number
from functools import wraps
import numpy as np
from llvmlite import ir
import numba
from numba import njit, typeof, objmode
from numba.core import cgutils, types, typing
from numba.core.pythonapi import box
from numba.core.errors import TypingError
from numba.core.registry import cpu_target
from numba.extending import (intrinsic, lower_builtin, overload_classmethod,
from numba.np import numpy_support
from numba.tests.support import TestCase, MemoryLeakMixin
@register_model(MyArrayType)
class MyArrayTypeModel(numba.core.datamodel.models.StructModel):

    def __init__(self, dmm, fe_type):
        ndim = fe_type.ndim
        members = [('meminfo', types.MemInfoPointer(fe_type.dtype)), ('parent', types.pyobject), ('nitems', types.intp), ('itemsize', types.intp), ('data', types.CPointer(fe_type.dtype)), ('shape', types.UniTuple(types.intp, ndim)), ('strides', types.UniTuple(types.intp, ndim)), ('extra_field', types.intp)]
        super(MyArrayTypeModel, self).__init__(dmm, fe_type, members)