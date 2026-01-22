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
class pdf_xrange(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    low = property(_mupdf.pdf_xrange_low_get, _mupdf.pdf_xrange_low_set)
    high = property(_mupdf.pdf_xrange_high_get, _mupdf.pdf_xrange_high_set)
    out = property(_mupdf.pdf_xrange_out_get, _mupdf.pdf_xrange_out_set)

    def __init__(self):
        _mupdf.pdf_xrange_swiginit(self, _mupdf.new_pdf_xrange())
    __swig_destroy__ = _mupdf.delete_pdf_xrange