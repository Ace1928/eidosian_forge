import warnings
from collections import Counter
from contextlib import nullcontext
from .._utils import set_module
from . import numeric as sb
from . import numerictypes as nt
from numpy.compat import os_fspath
from .arrayprint import _get_legacy_print_mode
def _createdtype(self, byteorder):
    dtype = sb.dtype({'names': self._names, 'formats': self._f_formats, 'offsets': self._offsets, 'titles': self._titles})
    if byteorder is not None:
        byteorder = _byteorderconv[byteorder[0]]
        dtype = dtype.newbyteorder(byteorder)
    self.dtype = dtype