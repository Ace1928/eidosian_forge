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
class fz_md5(object):
    """
    Structure definition is public to enable stack
    based allocation. Do not access the members directly.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    lo = property(_mupdf.fz_md5_lo_get, _mupdf.fz_md5_lo_set)
    hi = property(_mupdf.fz_md5_hi_get, _mupdf.fz_md5_hi_set)
    a = property(_mupdf.fz_md5_a_get, _mupdf.fz_md5_a_set)
    b = property(_mupdf.fz_md5_b_get, _mupdf.fz_md5_b_set)
    c = property(_mupdf.fz_md5_c_get, _mupdf.fz_md5_c_set)
    d = property(_mupdf.fz_md5_d_get, _mupdf.fz_md5_d_set)
    buffer = property(_mupdf.fz_md5_buffer_get, _mupdf.fz_md5_buffer_set)

    def __init__(self):
        _mupdf.fz_md5_swiginit(self, _mupdf.new_fz_md5())
    __swig_destroy__ = _mupdf.delete_fz_md5