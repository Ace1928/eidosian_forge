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
class fz_shade(object):
    """
    Structure is public to allow derived classes. Do not
    access the members directly.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    storable = property(_mupdf.fz_shade_storable_get, _mupdf.fz_shade_storable_set)
    bbox = property(_mupdf.fz_shade_bbox_get, _mupdf.fz_shade_bbox_set)
    colorspace = property(_mupdf.fz_shade_colorspace_get, _mupdf.fz_shade_colorspace_set)
    matrix = property(_mupdf.fz_shade_matrix_get, _mupdf.fz_shade_matrix_set)
    use_background = property(_mupdf.fz_shade_use_background_get, _mupdf.fz_shade_use_background_set)
    background = property(_mupdf.fz_shade_background_get, _mupdf.fz_shade_background_set)
    use_function = property(_mupdf.fz_shade_use_function_get, _mupdf.fz_shade_use_function_set)
    function = property(_mupdf.fz_shade_function_get, _mupdf.fz_shade_function_set)
    type = property(_mupdf.fz_shade_type_get, _mupdf.fz_shade_type_set)
    buffer = property(_mupdf.fz_shade_buffer_get, _mupdf.fz_shade_buffer_set)

    def __init__(self):
        _mupdf.fz_shade_swiginit(self, _mupdf.new_fz_shade())
    __swig_destroy__ = _mupdf.delete_fz_shade