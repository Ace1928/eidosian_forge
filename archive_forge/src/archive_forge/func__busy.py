from contextlib import contextmanager
import logging
import typing
import os
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import ffi_proxy
from rpy2.rinterface_lib import conversion
@ffi_proxy.callback(ffi_proxy._busy_def, openrlib._rinterface_cffi)
def _busy(which: int) -> None:
    try:
        busy(which)
    except Exception as e:
        logger.error(_BUSY_EXCEPTION_LOG, str(e))