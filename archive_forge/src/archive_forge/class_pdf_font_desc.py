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
class pdf_font_desc(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    storable = property(_mupdf.pdf_font_desc_storable_get, _mupdf.pdf_font_desc_storable_set)
    size = property(_mupdf.pdf_font_desc_size_get, _mupdf.pdf_font_desc_size_set)
    font = property(_mupdf.pdf_font_desc_font_get, _mupdf.pdf_font_desc_font_set)
    flags = property(_mupdf.pdf_font_desc_flags_get, _mupdf.pdf_font_desc_flags_set)
    italic_angle = property(_mupdf.pdf_font_desc_italic_angle_get, _mupdf.pdf_font_desc_italic_angle_set)
    ascent = property(_mupdf.pdf_font_desc_ascent_get, _mupdf.pdf_font_desc_ascent_set)
    descent = property(_mupdf.pdf_font_desc_descent_get, _mupdf.pdf_font_desc_descent_set)
    cap_height = property(_mupdf.pdf_font_desc_cap_height_get, _mupdf.pdf_font_desc_cap_height_set)
    x_height = property(_mupdf.pdf_font_desc_x_height_get, _mupdf.pdf_font_desc_x_height_set)
    missing_width = property(_mupdf.pdf_font_desc_missing_width_get, _mupdf.pdf_font_desc_missing_width_set)
    encoding = property(_mupdf.pdf_font_desc_encoding_get, _mupdf.pdf_font_desc_encoding_set)
    to_ttf_cmap = property(_mupdf.pdf_font_desc_to_ttf_cmap_get, _mupdf.pdf_font_desc_to_ttf_cmap_set)
    cid_to_gid_len = property(_mupdf.pdf_font_desc_cid_to_gid_len_get, _mupdf.pdf_font_desc_cid_to_gid_len_set)
    cid_to_gid = property(_mupdf.pdf_font_desc_cid_to_gid_get, _mupdf.pdf_font_desc_cid_to_gid_set)
    to_unicode = property(_mupdf.pdf_font_desc_to_unicode_get, _mupdf.pdf_font_desc_to_unicode_set)
    cid_to_ucs_len = property(_mupdf.pdf_font_desc_cid_to_ucs_len_get, _mupdf.pdf_font_desc_cid_to_ucs_len_set)
    cid_to_ucs = property(_mupdf.pdf_font_desc_cid_to_ucs_get, _mupdf.pdf_font_desc_cid_to_ucs_set)
    wmode = property(_mupdf.pdf_font_desc_wmode_get, _mupdf.pdf_font_desc_wmode_set)
    hmtx_len = property(_mupdf.pdf_font_desc_hmtx_len_get, _mupdf.pdf_font_desc_hmtx_len_set)
    hmtx_cap = property(_mupdf.pdf_font_desc_hmtx_cap_get, _mupdf.pdf_font_desc_hmtx_cap_set)
    dhmtx = property(_mupdf.pdf_font_desc_dhmtx_get, _mupdf.pdf_font_desc_dhmtx_set)
    hmtx = property(_mupdf.pdf_font_desc_hmtx_get, _mupdf.pdf_font_desc_hmtx_set)
    vmtx_len = property(_mupdf.pdf_font_desc_vmtx_len_get, _mupdf.pdf_font_desc_vmtx_len_set)
    vmtx_cap = property(_mupdf.pdf_font_desc_vmtx_cap_get, _mupdf.pdf_font_desc_vmtx_cap_set)
    dvmtx = property(_mupdf.pdf_font_desc_dvmtx_get, _mupdf.pdf_font_desc_dvmtx_set)
    vmtx = property(_mupdf.pdf_font_desc_vmtx_get, _mupdf.pdf_font_desc_vmtx_set)
    is_embedded = property(_mupdf.pdf_font_desc_is_embedded_get, _mupdf.pdf_font_desc_is_embedded_set)
    t3loading = property(_mupdf.pdf_font_desc_t3loading_get, _mupdf.pdf_font_desc_t3loading_set)

    def __init__(self):
        _mupdf.pdf_font_desc_swiginit(self, _mupdf.new_pdf_font_desc())
    __swig_destroy__ = _mupdf.delete_pdf_font_desc