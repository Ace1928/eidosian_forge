from contextlib import contextmanager
import logging
import typing
import os
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import ffi_proxy
from rpy2.rinterface_lib import conversion
@ffi_proxy.callback(ffi_proxy._yesnocancel_def, openrlib._rinterface_cffi)
def _yesnocancel(question):
    try:
        q = conversion._cchar_to_str(question, _CCHAR_ENCODING)
        res = yesnocancel(q)
    except Exception as e:
        logger.error(_YESNOCANCEL_EXCEPTION_LOG, str(e))
    return res