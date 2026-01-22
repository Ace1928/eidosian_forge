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
class pdf_mrange(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    low = property(_mupdf.pdf_mrange_low_get, _mupdf.pdf_mrange_low_set)
    out = property(_mupdf.pdf_mrange_out_get, _mupdf.pdf_mrange_out_set)

    def __init__(self):
        _mupdf.pdf_mrange_swiginit(self, _mupdf.new_pdf_mrange())
    __swig_destroy__ = _mupdf.delete_pdf_mrange