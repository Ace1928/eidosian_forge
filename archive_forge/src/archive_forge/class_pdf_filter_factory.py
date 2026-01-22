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
class pdf_filter_factory(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    filter = property(_mupdf.pdf_filter_factory_filter_get, _mupdf.pdf_filter_factory_filter_set)
    options = property(_mupdf.pdf_filter_factory_options_get, _mupdf.pdf_filter_factory_options_set)

    def __init__(self):
        _mupdf.pdf_filter_factory_swiginit(self, _mupdf.new_pdf_filter_factory())
    __swig_destroy__ = _mupdf.delete_pdf_filter_factory