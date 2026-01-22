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
class fz_default_colorspaces(object):
    """	Structure to hold default colorspaces."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    refs = property(_mupdf.fz_default_colorspaces_refs_get, _mupdf.fz_default_colorspaces_refs_set)
    gray = property(_mupdf.fz_default_colorspaces_gray_get, _mupdf.fz_default_colorspaces_gray_set)
    rgb = property(_mupdf.fz_default_colorspaces_rgb_get, _mupdf.fz_default_colorspaces_rgb_set)
    cmyk = property(_mupdf.fz_default_colorspaces_cmyk_get, _mupdf.fz_default_colorspaces_cmyk_set)
    oi = property(_mupdf.fz_default_colorspaces_oi_get, _mupdf.fz_default_colorspaces_oi_set)

    def __init__(self):
        _mupdf.fz_default_colorspaces_swiginit(self, _mupdf.new_fz_default_colorspaces())
    __swig_destroy__ = _mupdf.delete_fz_default_colorspaces