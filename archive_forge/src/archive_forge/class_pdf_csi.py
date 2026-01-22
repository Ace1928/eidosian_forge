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
class pdf_csi(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    doc = property(_mupdf.pdf_csi_doc_get, _mupdf.pdf_csi_doc_set)
    rdb = property(_mupdf.pdf_csi_rdb_get, _mupdf.pdf_csi_rdb_set)
    buf = property(_mupdf.pdf_csi_buf_get, _mupdf.pdf_csi_buf_set)
    cookie = property(_mupdf.pdf_csi_cookie_get, _mupdf.pdf_csi_cookie_set)
    gstate = property(_mupdf.pdf_csi_gstate_get, _mupdf.pdf_csi_gstate_set)
    xbalance = property(_mupdf.pdf_csi_xbalance_get, _mupdf.pdf_csi_xbalance_set)
    in_text = property(_mupdf.pdf_csi_in_text_get, _mupdf.pdf_csi_in_text_set)
    d1_rect = property(_mupdf.pdf_csi_d1_rect_get, _mupdf.pdf_csi_d1_rect_set)
    obj = property(_mupdf.pdf_csi_obj_get, _mupdf.pdf_csi_obj_set)
    name = property(_mupdf.pdf_csi_name_get, _mupdf.pdf_csi_name_set)
    string = property(_mupdf.pdf_csi_string_get, _mupdf.pdf_csi_string_set)
    string_len = property(_mupdf.pdf_csi_string_len_get, _mupdf.pdf_csi_string_len_set)
    top = property(_mupdf.pdf_csi_top_get, _mupdf.pdf_csi_top_set)
    stack = property(_mupdf.pdf_csi_stack_get, _mupdf.pdf_csi_stack_set)

    def __init__(self):
        _mupdf.pdf_csi_swiginit(self, _mupdf.new_pdf_csi())
    __swig_destroy__ = _mupdf.delete_pdf_csi