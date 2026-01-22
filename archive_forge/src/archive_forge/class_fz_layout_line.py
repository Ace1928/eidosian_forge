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
class fz_layout_line(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    x = property(_mupdf.fz_layout_line_x_get, _mupdf.fz_layout_line_x_set)
    y = property(_mupdf.fz_layout_line_y_get, _mupdf.fz_layout_line_y_set)
    font_size = property(_mupdf.fz_layout_line_font_size_get, _mupdf.fz_layout_line_font_size_set)
    p = property(_mupdf.fz_layout_line_p_get, _mupdf.fz_layout_line_p_set)
    text = property(_mupdf.fz_layout_line_text_get, _mupdf.fz_layout_line_text_set)
    next = property(_mupdf.fz_layout_line_next_get, _mupdf.fz_layout_line_next_set)

    def __init__(self):
        _mupdf.fz_layout_line_swiginit(self, _mupdf.new_fz_layout_line())
    __swig_destroy__ = _mupdf.delete_fz_layout_line