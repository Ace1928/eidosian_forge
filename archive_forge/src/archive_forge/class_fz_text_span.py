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
class fz_text_span(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    font = property(_mupdf.fz_text_span_font_get, _mupdf.fz_text_span_font_set)
    trm = property(_mupdf.fz_text_span_trm_get, _mupdf.fz_text_span_trm_set)
    wmode = property(_mupdf.fz_text_span_wmode_get, _mupdf.fz_text_span_wmode_set)
    bidi_level = property(_mupdf.fz_text_span_bidi_level_get, _mupdf.fz_text_span_bidi_level_set)
    markup_dir = property(_mupdf.fz_text_span_markup_dir_get, _mupdf.fz_text_span_markup_dir_set)
    language = property(_mupdf.fz_text_span_language_get, _mupdf.fz_text_span_language_set)
    len = property(_mupdf.fz_text_span_len_get, _mupdf.fz_text_span_len_set)
    cap = property(_mupdf.fz_text_span_cap_get, _mupdf.fz_text_span_cap_set)
    items = property(_mupdf.fz_text_span_items_get, _mupdf.fz_text_span_items_set)
    next = property(_mupdf.fz_text_span_next_get, _mupdf.fz_text_span_next_set)

    def __init__(self):
        _mupdf.fz_text_span_swiginit(self, _mupdf.new_fz_text_span())
    __swig_destroy__ = _mupdf.delete_fz_text_span