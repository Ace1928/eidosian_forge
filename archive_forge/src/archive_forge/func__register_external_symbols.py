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
def _register_external_symbols() -> None:
    python_cchar = ffi.new('char []', b'.Python')
    ffi_proxy = openrlib.ffi_proxy
    if ffi_proxy.get_ffi_mode(openrlib._rinterface_cffi) == ffi_proxy.InterfaceType.ABI:
        python_callback = ffi.cast('DL_FUNC', _evaluate_in_r)
    else:
        python_callback = ffi.cast('DL_FUNC', openrlib.rlib._evaluate_in_r)
    externalmethods = ffi.new('R_ExternalMethodDef[]', [[python_cchar, python_callback, -1], [ffi.NULL, ffi.NULL, 0]])
    openrlib.rlib.R_registerRoutines(openrlib.rlib.R_getEmbeddingDllInfo(), ffi.NULL, ffi.NULL, ffi.NULL, externalmethods)