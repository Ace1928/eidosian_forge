from sys import version_info as _swig_python_version_info
import weakref
import inspect
import os
import re
import sys
import traceback
import inspect
import io
import os
import sys
import traceback
import types
class fz_sha256(object):
    """
    Structure definition is public to enable stack
    based allocation. Do not access the members directly.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    state = property(_mupdf.fz_sha256_state_get, _mupdf.fz_sha256_state_set)
    count = property(_mupdf.fz_sha256_count_get, _mupdf.fz_sha256_count_set)

    def __init__(self):
        _mupdf.fz_sha256_swiginit(self, _mupdf.new_fz_sha256())
    __swig_destroy__ = _mupdf.delete_fz_sha256