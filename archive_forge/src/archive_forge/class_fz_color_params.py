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
class fz_color_params(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    ri = property(_mupdf.fz_color_params_ri_get, _mupdf.fz_color_params_ri_set)
    bp = property(_mupdf.fz_color_params_bp_get, _mupdf.fz_color_params_bp_set)
    op = property(_mupdf.fz_color_params_op_get, _mupdf.fz_color_params_op_set)
    opm = property(_mupdf.fz_color_params_opm_get, _mupdf.fz_color_params_opm_set)

    def __init__(self):
        _mupdf.fz_color_params_swiginit(self, _mupdf.new_fz_color_params())
    __swig_destroy__ = _mupdf.delete_fz_color_params