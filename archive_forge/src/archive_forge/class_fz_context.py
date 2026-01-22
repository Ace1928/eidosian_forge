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
class fz_context(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    user = property(_mupdf.fz_context_user_get, _mupdf.fz_context_user_set)
    alloc = property(_mupdf.fz_context_alloc_get, _mupdf.fz_context_alloc_set)
    locks = property(_mupdf.fz_context_locks_get, _mupdf.fz_context_locks_set)
    error = property(_mupdf.fz_context_error_get, _mupdf.fz_context_error_set)
    warn = property(_mupdf.fz_context_warn_get, _mupdf.fz_context_warn_set)
    aa = property(_mupdf.fz_context_aa_get, _mupdf.fz_context_aa_set)
    seed48 = property(_mupdf.fz_context_seed48_get, _mupdf.fz_context_seed48_set)
    icc_enabled = property(_mupdf.fz_context_icc_enabled_get, _mupdf.fz_context_icc_enabled_set)
    throw_on_repair = property(_mupdf.fz_context_throw_on_repair_get, _mupdf.fz_context_throw_on_repair_set)
    handler = property(_mupdf.fz_context_handler_get, _mupdf.fz_context_handler_set)
    archive = property(_mupdf.fz_context_archive_get, _mupdf.fz_context_archive_set)
    style = property(_mupdf.fz_context_style_get, _mupdf.fz_context_style_set)
    tuning = property(_mupdf.fz_context_tuning_get, _mupdf.fz_context_tuning_set)
    stddbg = property(_mupdf.fz_context_stddbg_get, _mupdf.fz_context_stddbg_set)
    font = property(_mupdf.fz_context_font_get, _mupdf.fz_context_font_set)
    colorspace = property(_mupdf.fz_context_colorspace_get, _mupdf.fz_context_colorspace_set)
    store = property(_mupdf.fz_context_store_get, _mupdf.fz_context_store_set)
    glyph_cache = property(_mupdf.fz_context_glyph_cache_get, _mupdf.fz_context_glyph_cache_set)

    def __init__(self):
        _mupdf.fz_context_swiginit(self, _mupdf.new_fz_context())
    __swig_destroy__ = _mupdf.delete_fz_context