from contextlib import contextmanager
import logging
import typing
import os
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import ffi_proxy
from rpy2.rinterface_lib import conversion
@ffi_proxy.callback(ffi_proxy._callback_def, openrlib._rinterface_cffi)
def _callback() -> None:
    try:
        callback()
    except Exception as e:
        logger.error(_CALLBACK_EXCEPTION_LOG, str(e))