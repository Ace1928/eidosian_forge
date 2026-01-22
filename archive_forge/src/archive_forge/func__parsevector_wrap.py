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
@ffi_proxy.callback(ffi_proxy._parsevector_wrap_def, openrlib._rinterface_cffi)
def _parsevector_wrap(data: FFI.CData):
    try:
        cdata, num, status = ffi.from_handle(data)
        res = openrlib.rlib.R_ParseVector(cdata, num, status, openrlib.rlib.R_NilValue)
    except Exception as e:
        res = openrlib.rlib.R_NilValue
        logger.error('_parsevector_wrap: %s', str(e))
    return res