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
@ffi_proxy.callback(ffi_proxy._exec_findvar_in_frame_def, openrlib._rinterface_cffi)
def _exec_findvar_in_frame(cdata):
    cdata_struct = openrlib.ffi.cast('struct RPY2_sym_env_data *', cdata)
    res = openrlib.rlib.Rf_findVarInFrame(cdata_struct.environment, cdata_struct.symbol)
    cdata_struct.data = res
    return