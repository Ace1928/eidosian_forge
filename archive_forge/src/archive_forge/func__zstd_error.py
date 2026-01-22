from __future__ import absolute_import, unicode_literals
import io
import os
from ._cffi import (  # type: ignore
def _zstd_error(zresult):
    return ffi.string(lib.ZSTD_getErrorName(zresult)).decode('utf-8')