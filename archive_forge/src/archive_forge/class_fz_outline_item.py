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
class fz_outline_item(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    title = property(_mupdf.fz_outline_item_title_get, _mupdf.fz_outline_item_title_set)
    uri = property(_mupdf.fz_outline_item_uri_get, _mupdf.fz_outline_item_uri_set)
    is_open = property(_mupdf.fz_outline_item_is_open_get, _mupdf.fz_outline_item_is_open_set)

    def __init__(self):
        _mupdf.fz_outline_item_swiginit(self, _mupdf.new_fz_outline_item())
    __swig_destroy__ = _mupdf.delete_fz_outline_item