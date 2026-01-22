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
class fz_link_dest(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    loc = property(_mupdf.fz_link_dest_loc_get, _mupdf.fz_link_dest_loc_set)
    type = property(_mupdf.fz_link_dest_type_get, _mupdf.fz_link_dest_type_set)
    x = property(_mupdf.fz_link_dest_x_get, _mupdf.fz_link_dest_x_set)
    y = property(_mupdf.fz_link_dest_y_get, _mupdf.fz_link_dest_y_set)
    w = property(_mupdf.fz_link_dest_w_get, _mupdf.fz_link_dest_w_set)
    h = property(_mupdf.fz_link_dest_h_get, _mupdf.fz_link_dest_h_set)
    zoom = property(_mupdf.fz_link_dest_zoom_get, _mupdf.fz_link_dest_zoom_set)

    def __init__(self):
        _mupdf.fz_link_dest_swiginit(self, _mupdf.new_fz_link_dest())
    __swig_destroy__ = _mupdf.delete_fz_link_dest