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
class pdf_hint_shared(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    number = property(_mupdf.pdf_hint_shared_number_get, _mupdf.pdf_hint_shared_number_set)
    offset = property(_mupdf.pdf_hint_shared_offset_get, _mupdf.pdf_hint_shared_offset_set)

    def __init__(self):
        _mupdf.pdf_hint_shared_swiginit(self, _mupdf.new_pdf_hint_shared())
    __swig_destroy__ = _mupdf.delete_pdf_hint_shared