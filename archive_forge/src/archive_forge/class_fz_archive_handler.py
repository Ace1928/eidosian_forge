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
class fz_archive_handler(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    recognize = property(_mupdf.fz_archive_handler_recognize_get, _mupdf.fz_archive_handler_recognize_set)
    open = property(_mupdf.fz_archive_handler_open_get, _mupdf.fz_archive_handler_open_set)

    def __init__(self):
        _mupdf.fz_archive_handler_swiginit(self, _mupdf.new_fz_archive_handler())
    __swig_destroy__ = _mupdf.delete_fz_archive_handler