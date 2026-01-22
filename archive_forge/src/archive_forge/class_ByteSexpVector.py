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
class ByteSexpVector(SexpVectorWithNumpyInterface):
    """Array of bytes.

    This is the R equivalent to a Python :class:`bytesarray`.
    """
    _R_TYPE = openrlib.rlib.RAWSXP
    _R_SIZEOF_ELT = _rinterface.ffi.sizeof('char')
    _NP_TYPESTR = '|u1'
    _R_GET_PTR = staticmethod(openrlib.RAW)

    @staticmethod
    def _CAST_IN(x: typing.Any) -> int:
        if isinstance(x, int):
            if x > 255:
                raise ValueError('byte must be in range(0, 256)')
            res = x
        elif isinstance(x, (bytes, bytearray)):
            if len(x) != 1:
                raise ValueError('byte must be a single character')
            res = ord(x)
        else:
            raise ValueError('byte must be an integer [0, 255] or a single byte character')
        return res

    @staticmethod
    def _R_VECTOR_ELT(x, i: int) -> None:
        return openrlib.RAW(x)[i]

    @staticmethod
    def _R_SET_VECTOR_ELT(x, i: int, val) -> None:
        openrlib.RAW(x)[i] = val

    def __getitem__(self, i: Union[int, slice]) -> Union[int, 'ByteSexpVector']:
        cdata = self.__sexp__._cdata
        if isinstance(i, int):
            i_c = _rinterface._python_index_to_c(cdata, i)
            res = openrlib.RAW_ELT(cdata, i_c)
        elif isinstance(i, slice):
            res = type(self).from_iterable([openrlib.RAW_ELT(cdata, i_c) for i_c in range(*i.indices(len(self)))])
        else:
            raise TypeError('Indices must be integers or slices, not %s' % type(i))
        return res

    def __setitem__(self, i: Union[int, slice], value) -> None:
        cdata = self.__sexp__._cdata
        if isinstance(i, int):
            i_c = _rinterface._python_index_to_c(cdata, i)
            openrlib.RAW(cdata)[i_c] = self._CAST_IN(value)
        elif isinstance(i, slice):
            for i_c, v in zip(range(*i.indices(len(self))), value):
                if v > 255:
                    raise ValueError('byte must be in range(0, 256)')
                openrlib.RAW(cdata)[i_c] = self._CAST_IN(v)
        else:
            raise TypeError('Indices must be integers or slices, not %s' % type(i))