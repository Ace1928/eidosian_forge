import io
import os
import numpy as np
import scipy.sparse
from scipy.io import _mmio
def _get_write_cursor(target, h=None, comment=None, parallelism=None, symmetry='general', precision=None):
    """
    Open file for writing.
    """
    from . import _fmm_core
    if parallelism is None:
        parallelism = PARALLELISM
    if comment is None:
        comment = ''
    if symmetry is None:
        symmetry = 'general'
    if precision is None:
        precision = -1
    if not h:
        h = _fmm_core.header(comment=comment, symmetry=symmetry)
    try:
        target = os.fspath(target)
        return _fmm_core.open_write_file(str(target), h, parallelism, precision)
    except TypeError:
        pass
    if hasattr(target, 'write'):
        if isinstance(target, io.TextIOBase):
            raise TypeError('target stream must be open in binary mode.')
        return _fmm_core.open_write_stream(target, h, parallelism, precision)
    else:
        raise TypeError('Unknown source object')