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
class pdf_mark_list(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    len = property(_mupdf.pdf_mark_list_len_get, _mupdf.pdf_mark_list_len_set)
    max = property(_mupdf.pdf_mark_list_max_get, _mupdf.pdf_mark_list_max_set)
    list = property(_mupdf.pdf_mark_list_list_get, _mupdf.pdf_mark_list_list_set)
    local_list = property(_mupdf.pdf_mark_list_local_list_get, _mupdf.pdf_mark_list_local_list_set)

    def __init__(self):
        _mupdf.pdf_mark_list_swiginit(self, _mupdf.new_pdf_mark_list())
    __swig_destroy__ = _mupdf.delete_pdf_mark_list