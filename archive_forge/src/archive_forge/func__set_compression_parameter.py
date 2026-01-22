from __future__ import absolute_import, unicode_literals
import io
import os
from ._cffi import (  # type: ignore
def _set_compression_parameter(params, param, value):
    zresult = lib.ZSTD_CCtxParams_setParameter(params, param, value)
    if lib.ZSTD_isError(zresult):
        raise ZstdError('unable to set compression context parameter: %s' % _zstd_error(zresult))