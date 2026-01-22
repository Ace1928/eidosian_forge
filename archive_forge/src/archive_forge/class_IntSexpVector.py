import abc
import atexit
import contextlib
import contextvars
import csv
import enum
import functools
import inspect
import os
import math
import platform
import signal
import subprocess
import textwrap
import threading
import typing
import warnings
from typing import Union
from rpy2.rinterface_lib import openrlib
import rpy2.rinterface_lib._rinterface_capi as _rinterface
import rpy2.rinterface_lib.embedded as embedded
import rpy2.rinterface_lib.conversion as conversion
from rpy2.rinterface_lib.conversion import _cdata_res_to_rinterface
import rpy2.rinterface_lib.memorymanagement as memorymanagement
from rpy2.rinterface_lib import na_values
from rpy2.rinterface_lib.sexp import NULL
from rpy2.rinterface_lib.sexp import NULLType
import rpy2.rinterface_lib.bufferprotocol as bufferprotocol
from rpy2.rinterface_lib import sexp
from rpy2.rinterface_lib.sexp import CharSexp  # noqa: F401
from rpy2.rinterface_lib.sexp import RTYPES
from rpy2.rinterface_lib.sexp import SexpVector
from rpy2.rinterface_lib.sexp import StrSexpVector
from rpy2.rinterface_lib.sexp import Sexp
from rpy2.rinterface_lib.sexp import SexpEnvironment
from rpy2.rinterface_lib.sexp import unserialize  # noqa: F401
from rpy2.rinterface_lib.sexp import emptyenv
from rpy2.rinterface_lib.sexp import baseenv
from rpy2.rinterface_lib.sexp import globalenv
class IntSexpVector(SexpVectorWithNumpyInterface):
    _R_TYPE = openrlib.rlib.INTSXP
    _R_SET_VECTOR_ELT = openrlib.SET_INTEGER_ELT
    _R_VECTOR_ELT = openrlib.INTEGER_ELT
    _R_SIZEOF_ELT = _rinterface.ffi.sizeof('int')
    _NP_TYPESTR = '|i'
    _R_GET_PTR = staticmethod(openrlib.INTEGER)
    _CAST_IN = staticmethod(nullable_int)

    def __getitem__(self, i: Union[int, slice]) -> Union[int, 'IntSexpVector']:
        cdata = self.__sexp__._cdata
        if isinstance(i, int):
            i_c = _rinterface._python_index_to_c(cdata, i)
            res = openrlib.INTEGER_ELT(cdata, i_c)
            if res == NA_Integer:
                res = NA_Integer
        elif isinstance(i, slice):
            res = type(self).from_iterable([openrlib.INTEGER_ELT(cdata, i_c) for i_c in range(*i.indices(len(self)))])
        else:
            raise TypeError('Indices must be integers or slices, not %s' % type(i))
        return res

    def __setitem__(self, i: Union[int, slice], value) -> None:
        cdata = self.__sexp__._cdata
        if isinstance(i, int):
            i_c = _rinterface._python_index_to_c(cdata, i)
            openrlib.SET_INTEGER_ELT(cdata, i_c, int(value))
        elif isinstance(i, slice):
            for i_c, v in zip(range(*i.indices(len(self))), value):
                openrlib.SET_INTEGER_ELT(cdata, i_c, int(v))
        else:
            raise TypeError('Indices must be integers or slices, not %s' % type(i))

    def memoryview(self) -> memoryview:
        return vector_memoryview(self, 'int', 'i')