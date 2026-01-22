from contextlib import contextmanager
import logging
import typing
import os
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import ffi_proxy
from rpy2.rinterface_lib import conversion
@ffi_proxy.callback(ffi_proxy._showfiles_def, openrlib._rinterface_cffi)
def _showfiles(nfiles: int, files, headers, wtitle, delete, pager) -> int:
    filenames = []
    headers_str = []
    wtitle_str = None
    pager_str = None
    try:
        wtitle_str = conversion._cchar_to_str(wtitle, _CCHAR_ENCODING)
        pager_str = conversion._cchar_to_str(pager, _CCHAR_ENCODING)
        for i in range(nfiles):
            filenames.append(conversion._cchar_to_str(files[i], _CCHAR_ENCODING))
            headers_str.append(conversion._cchar_to_str(headers[i], _CCHAR_ENCODING))
    except Exception as e:
        logger.error(_SHOWFILE_INTERNAL_EXCEPTION_LOG, str(e))
    if len(filenames):
        res = 0
    else:
        res = 1
    try:
        showfiles(tuple(filenames), tuple(headers_str), wtitle_str, pager_str)
    except Exception as e:
        res = 1
        logger.error(_SHOWFILE_EXCEPTION_LOG, str(e))
    return res