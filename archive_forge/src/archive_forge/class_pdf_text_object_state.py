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
class pdf_text_object_state(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    text = property(_mupdf.pdf_text_object_state_text_get, _mupdf.pdf_text_object_state_text_set)
    text_bbox = property(_mupdf.pdf_text_object_state_text_bbox_get, _mupdf.pdf_text_object_state_text_bbox_set)
    tlm = property(_mupdf.pdf_text_object_state_tlm_get, _mupdf.pdf_text_object_state_tlm_set)
    tm = property(_mupdf.pdf_text_object_state_tm_get, _mupdf.pdf_text_object_state_tm_set)
    text_mode = property(_mupdf.pdf_text_object_state_text_mode_get, _mupdf.pdf_text_object_state_text_mode_set)
    cid = property(_mupdf.pdf_text_object_state_cid_get, _mupdf.pdf_text_object_state_cid_set)
    gid = property(_mupdf.pdf_text_object_state_gid_get, _mupdf.pdf_text_object_state_gid_set)
    char_bbox = property(_mupdf.pdf_text_object_state_char_bbox_get, _mupdf.pdf_text_object_state_char_bbox_set)
    fontdesc = property(_mupdf.pdf_text_object_state_fontdesc_get, _mupdf.pdf_text_object_state_fontdesc_set)
    char_tx = property(_mupdf.pdf_text_object_state_char_tx_get, _mupdf.pdf_text_object_state_char_tx_set)
    char_ty = property(_mupdf.pdf_text_object_state_char_ty_get, _mupdf.pdf_text_object_state_char_ty_set)

    def __init__(self):
        _mupdf.pdf_text_object_state_swiginit(self, _mupdf.new_pdf_text_object_state())
    __swig_destroy__ = _mupdf.delete_pdf_text_object_state