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
class pdf_redact_options(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    black_boxes = property(_mupdf.pdf_redact_options_black_boxes_get, _mupdf.pdf_redact_options_black_boxes_set)
    image_method = property(_mupdf.pdf_redact_options_image_method_get, _mupdf.pdf_redact_options_image_method_set)
    line_art = property(_mupdf.pdf_redact_options_line_art_get, _mupdf.pdf_redact_options_line_art_set)

    def __init__(self):
        _mupdf.pdf_redact_options_swiginit(self, _mupdf.new_pdf_redact_options())
    __swig_destroy__ = _mupdf.delete_pdf_redact_options