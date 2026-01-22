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
class pdf_sanitize_filter_options(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    opaque = property(_mupdf.pdf_sanitize_filter_options_opaque_get, _mupdf.pdf_sanitize_filter_options_opaque_set)
    image_filter = property(_mupdf.pdf_sanitize_filter_options_image_filter_get, _mupdf.pdf_sanitize_filter_options_image_filter_set)
    text_filter = property(_mupdf.pdf_sanitize_filter_options_text_filter_get, _mupdf.pdf_sanitize_filter_options_text_filter_set)
    after_text_object = property(_mupdf.pdf_sanitize_filter_options_after_text_object_get, _mupdf.pdf_sanitize_filter_options_after_text_object_set)
    culler = property(_mupdf.pdf_sanitize_filter_options_culler_get, _mupdf.pdf_sanitize_filter_options_culler_set)

    def __init__(self):
        _mupdf.pdf_sanitize_filter_options_swiginit(self, _mupdf.new_pdf_sanitize_filter_options())
    __swig_destroy__ = _mupdf.delete_pdf_sanitize_filter_options