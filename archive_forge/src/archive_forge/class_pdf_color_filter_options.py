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
class pdf_color_filter_options(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    opaque = property(_mupdf.pdf_color_filter_options_opaque_get, _mupdf.pdf_color_filter_options_opaque_set)
    color_rewrite = property(_mupdf.pdf_color_filter_options_color_rewrite_get, _mupdf.pdf_color_filter_options_color_rewrite_set)
    image_rewrite = property(_mupdf.pdf_color_filter_options_image_rewrite_get, _mupdf.pdf_color_filter_options_image_rewrite_set)
    shade_rewrite = property(_mupdf.pdf_color_filter_options_shade_rewrite_get, _mupdf.pdf_color_filter_options_shade_rewrite_set)
    repeated_image_rewrite = property(_mupdf.pdf_color_filter_options_repeated_image_rewrite_get, _mupdf.pdf_color_filter_options_repeated_image_rewrite_set)

    def __init__(self):
        _mupdf.pdf_color_filter_options_swiginit(self, _mupdf.new_pdf_color_filter_options())
    __swig_destroy__ = _mupdf.delete_pdf_color_filter_options