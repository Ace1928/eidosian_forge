import numpy
import warnings
from numpy.lib.utils import safe_eval, drop_metadata
from numpy.compat import (
def _wrap_header_guess_version(header):
    """
    Like `_wrap_header`, but chooses an appropriate version given the contents
    """
    try:
        return _wrap_header(header, (1, 0))
    except ValueError:
        pass
    try:
        ret = _wrap_header(header, (2, 0))
    except UnicodeEncodeError:
        pass
    else:
        warnings.warn('Stored array in format 2.0. It can only beread by NumPy >= 1.9', UserWarning, stacklevel=2)
        return ret
    header = _wrap_header(header, (3, 0))
    warnings.warn('Stored array in format 3.0. It can only be read by NumPy >= 1.17', UserWarning, stacklevel=2)
    return header