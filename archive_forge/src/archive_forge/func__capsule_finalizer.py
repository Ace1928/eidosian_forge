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
@ffi_proxy.callback(ffi_proxy._capsule_finalizer_def, openrlib._rinterface_cffi)
def _capsule_finalizer(cdata: FFI.CData) -> None:
    try:
        openrlib.rlib.R_ClearExternalPtr(cdata)
    except Exception as e:
        warnings.warn('Exception downgraded to warning: %s' % str(e))