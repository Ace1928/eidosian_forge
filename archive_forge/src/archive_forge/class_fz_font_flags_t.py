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
class fz_font_flags_t(object):
    """
    Every fz_font carries a set of flags
    within it, in a fz_font_flags_t structure.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    is_mono = property(_mupdf.fz_font_flags_t_is_mono_get, _mupdf.fz_font_flags_t_is_mono_set)
    is_serif = property(_mupdf.fz_font_flags_t_is_serif_get, _mupdf.fz_font_flags_t_is_serif_set)
    is_bold = property(_mupdf.fz_font_flags_t_is_bold_get, _mupdf.fz_font_flags_t_is_bold_set)
    is_italic = property(_mupdf.fz_font_flags_t_is_italic_get, _mupdf.fz_font_flags_t_is_italic_set)
    ft_substitute = property(_mupdf.fz_font_flags_t_ft_substitute_get, _mupdf.fz_font_flags_t_ft_substitute_set)
    ft_stretch = property(_mupdf.fz_font_flags_t_ft_stretch_get, _mupdf.fz_font_flags_t_ft_stretch_set)
    fake_bold = property(_mupdf.fz_font_flags_t_fake_bold_get, _mupdf.fz_font_flags_t_fake_bold_set)
    fake_italic = property(_mupdf.fz_font_flags_t_fake_italic_get, _mupdf.fz_font_flags_t_fake_italic_set)
    has_opentype = property(_mupdf.fz_font_flags_t_has_opentype_get, _mupdf.fz_font_flags_t_has_opentype_set)
    invalid_bbox = property(_mupdf.fz_font_flags_t_invalid_bbox_get, _mupdf.fz_font_flags_t_invalid_bbox_set)
    cjk = property(_mupdf.fz_font_flags_t_cjk_get, _mupdf.fz_font_flags_t_cjk_set)
    cjk_lang = property(_mupdf.fz_font_flags_t_cjk_lang_get, _mupdf.fz_font_flags_t_cjk_lang_set)
    embed = property(_mupdf.fz_font_flags_t_embed_get, _mupdf.fz_font_flags_t_embed_set)
    never_embed = property(_mupdf.fz_font_flags_t_never_embed_get, _mupdf.fz_font_flags_t_never_embed_set)

    def __init__(self):
        _mupdf.fz_font_flags_t_swiginit(self, _mupdf.new_fz_font_flags_t())
    __swig_destroy__ = _mupdf.delete_fz_font_flags_t