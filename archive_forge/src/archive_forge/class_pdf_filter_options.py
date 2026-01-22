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
class pdf_filter_options(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    recurse = property(_mupdf.pdf_filter_options_recurse_get, _mupdf.pdf_filter_options_recurse_set)
    instance_forms = property(_mupdf.pdf_filter_options_instance_forms_get, _mupdf.pdf_filter_options_instance_forms_set)
    ascii = property(_mupdf.pdf_filter_options_ascii_get, _mupdf.pdf_filter_options_ascii_set)
    no_update = property(_mupdf.pdf_filter_options_no_update_get, _mupdf.pdf_filter_options_no_update_set)
    opaque = property(_mupdf.pdf_filter_options_opaque_get, _mupdf.pdf_filter_options_opaque_set)
    complete = property(_mupdf.pdf_filter_options_complete_get, _mupdf.pdf_filter_options_complete_set)
    filters = property(_mupdf.pdf_filter_options_filters_get, _mupdf.pdf_filter_options_filters_set)
    newlines = property(_mupdf.pdf_filter_options_newlines_get, _mupdf.pdf_filter_options_newlines_set)

    def __init__(self):
        _mupdf.pdf_filter_options_swiginit(self, _mupdf.new_pdf_filter_options())
    __swig_destroy__ = _mupdf.delete_pdf_filter_options