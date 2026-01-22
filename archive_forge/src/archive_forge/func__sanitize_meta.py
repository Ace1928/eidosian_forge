import datetime
from ..core import Format
from ..core.request import URI_BYTES, URI_FILE
import numpy as np
import warnings
@staticmethod
def _sanitize_meta(meta):
    ret = {}
    for key, value in meta.items():
        if key in WRITE_METADATA_KEYS:
            if key == 'predictor' and (not isinstance(value, bool)):
                ret[key] = value > 1
            elif key == 'compress' and value != 0:
                warnings.warn('The use of `compress` is deprecated. Use `compression` and `compressionargs` instead.', DeprecationWarning)
                if _tifffile.__version__ < '2022':
                    ret['compression'] = (8, value)
                else:
                    ret['compression'] = 'zlib'
                    ret['compressionargs'] = {'level': value}
            else:
                ret[key] = value
    return ret