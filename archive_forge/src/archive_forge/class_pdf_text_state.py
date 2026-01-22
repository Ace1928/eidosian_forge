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
class pdf_text_state(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    char_space = property(_mupdf.pdf_text_state_char_space_get, _mupdf.pdf_text_state_char_space_set)
    word_space = property(_mupdf.pdf_text_state_word_space_get, _mupdf.pdf_text_state_word_space_set)
    scale = property(_mupdf.pdf_text_state_scale_get, _mupdf.pdf_text_state_scale_set)
    leading = property(_mupdf.pdf_text_state_leading_get, _mupdf.pdf_text_state_leading_set)
    font = property(_mupdf.pdf_text_state_font_get, _mupdf.pdf_text_state_font_set)
    fontname = property(_mupdf.pdf_text_state_fontname_get, _mupdf.pdf_text_state_fontname_set)
    size = property(_mupdf.pdf_text_state_size_get, _mupdf.pdf_text_state_size_set)
    render = property(_mupdf.pdf_text_state_render_get, _mupdf.pdf_text_state_render_set)
    rise = property(_mupdf.pdf_text_state_rise_get, _mupdf.pdf_text_state_rise_set)

    def __init__(self):
        _mupdf.pdf_text_state_swiginit(self, _mupdf.new_pdf_text_state())
    __swig_destroy__ = _mupdf.delete_pdf_text_state