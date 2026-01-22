import abc
import enum
import inspect
import logging
from typing import Tuple
import typing
import warnings
from rpy2.rinterface_lib import ffi_proxy
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import conversion
from rpy2.rinterface_lib import embedded
from rpy2.rinterface_lib import memorymanagement
from _cffi_backend import FFI  # type: ignore
def _string_setitem(cdata: FFI.CData, i: int, CE: int, value_b) -> None:
    rlib = openrlib.rlib
    rlib.SET_STRING_ELT(cdata, i, rlib.Rf_mkCharCE(value_b, CE))