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
class fz_layout_char(object):
    """	Simple text layout (for use with annotation editing primarily)."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    x = property(_mupdf.fz_layout_char_x_get, _mupdf.fz_layout_char_x_set)
    advance = property(_mupdf.fz_layout_char_advance_get, _mupdf.fz_layout_char_advance_set)
    p = property(_mupdf.fz_layout_char_p_get, _mupdf.fz_layout_char_p_set)
    next = property(_mupdf.fz_layout_char_next_get, _mupdf.fz_layout_char_next_set)

    def __init__(self):
        _mupdf.fz_layout_char_swiginit(self, _mupdf.new_fz_layout_char())
    __swig_destroy__ = _mupdf.delete_fz_layout_char