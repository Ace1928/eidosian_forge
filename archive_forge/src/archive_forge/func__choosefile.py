from contextlib import contextmanager
import logging
import typing
import os
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import ffi_proxy
from rpy2.rinterface_lib import conversion
@ffi_proxy.callback(ffi_proxy._choosefile_def, openrlib._rinterface_cffi)
def _choosefile(new, buf, n: int) -> int:
    try:
        res = choosefile(new)
    except Exception as e:
        logger.error(_CHOOSEFILE_EXCEPTION_LOG, str(e))
        res = None
    if res is None:
        return 0
    res_cdata = conversion._str_to_cchar(res)
    openrlib.ffi.memmove(buf, res_cdata, len(res_cdata))
    return len(res_cdata)