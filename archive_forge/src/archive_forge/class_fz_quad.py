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
class fz_quad(object):
    """
    A representation for a region defined by 4 points.

    The significant difference between quads and rects is that
    the edges of quads are not axis aligned.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    ul = property(_mupdf.fz_quad_ul_get, _mupdf.fz_quad_ul_set)
    ur = property(_mupdf.fz_quad_ur_get, _mupdf.fz_quad_ur_set)
    ll = property(_mupdf.fz_quad_ll_get, _mupdf.fz_quad_ll_set)
    lr = property(_mupdf.fz_quad_lr_get, _mupdf.fz_quad_lr_set)

    def __init__(self):
        _mupdf.fz_quad_swiginit(self, _mupdf.new_fz_quad())
    __swig_destroy__ = _mupdf.delete_fz_quad