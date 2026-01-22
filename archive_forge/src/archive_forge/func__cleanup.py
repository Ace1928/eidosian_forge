from contextlib import contextmanager
import logging
import typing
import os
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import ffi_proxy
from rpy2.rinterface_lib import conversion
@ffi_proxy.callback(ffi_proxy._cleanup_def, openrlib._rinterface_cffi)
def _cleanup(saveact, status, runlast):
    try:
        cleanup(saveact, status, runlast)
    except Exception as e:
        logger.error(_CLEANUP_EXCEPTION_LOG, str(e))