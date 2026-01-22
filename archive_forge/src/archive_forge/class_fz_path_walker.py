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
class fz_path_walker(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    moveto = property(_mupdf.fz_path_walker_moveto_get, _mupdf.fz_path_walker_moveto_set)
    lineto = property(_mupdf.fz_path_walker_lineto_get, _mupdf.fz_path_walker_lineto_set)
    curveto = property(_mupdf.fz_path_walker_curveto_get, _mupdf.fz_path_walker_curveto_set)
    closepath = property(_mupdf.fz_path_walker_closepath_get, _mupdf.fz_path_walker_closepath_set)
    quadto = property(_mupdf.fz_path_walker_quadto_get, _mupdf.fz_path_walker_quadto_set)
    curvetov = property(_mupdf.fz_path_walker_curvetov_get, _mupdf.fz_path_walker_curvetov_set)
    curvetoy = property(_mupdf.fz_path_walker_curvetoy_get, _mupdf.fz_path_walker_curvetoy_set)
    rectto = property(_mupdf.fz_path_walker_rectto_get, _mupdf.fz_path_walker_rectto_set)

    def __init__(self):
        _mupdf.fz_path_walker_swiginit(self, _mupdf.new_fz_path_walker())
    __swig_destroy__ = _mupdf.delete_fz_path_walker