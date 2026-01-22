from contextlib import contextmanager
import logging
import typing
import os
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import ffi_proxy
from rpy2.rinterface_lib import conversion
@ffi_proxy.callback(ffi_proxy._consolewrite_ex_def, openrlib._rinterface_cffi)
def _consolewrite_ex(buf, n: int, otype: int) -> None:
    s = conversion._cchar_to_str_with_maxlen(buf, n, _CCHAR_ENCODING)
    try:
        if otype == 0:
            consolewrite_print(s)
        else:
            consolewrite_warnerror(s)
    except Exception as e:
        logger.error(_WRITECONSOLE_EXCEPTION_LOG, str(e))