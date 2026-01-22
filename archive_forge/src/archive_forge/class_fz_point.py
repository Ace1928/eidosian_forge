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
class fz_point(object):
    """	fz_point is a point in a two-dimensional space."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    x = property(_mupdf.fz_point_x_get, _mupdf.fz_point_x_set)
    y = property(_mupdf.fz_point_y_get, _mupdf.fz_point_y_set)

    def __init__(self):
        _mupdf.fz_point_swiginit(self, _mupdf.new_fz_point())
    __swig_destroy__ = _mupdf.delete_fz_point