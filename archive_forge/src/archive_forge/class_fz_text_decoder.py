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
class fz_text_decoder(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    decode_bound = property(_mupdf.fz_text_decoder_decode_bound_get, _mupdf.fz_text_decoder_decode_bound_set)
    decode_size = property(_mupdf.fz_text_decoder_decode_size_get, _mupdf.fz_text_decoder_decode_size_set)
    decode = property(_mupdf.fz_text_decoder_decode_get, _mupdf.fz_text_decoder_decode_set)
    table1 = property(_mupdf.fz_text_decoder_table1_get, _mupdf.fz_text_decoder_table1_set)
    table2 = property(_mupdf.fz_text_decoder_table2_get, _mupdf.fz_text_decoder_table2_set)

    def __init__(self):
        _mupdf.fz_text_decoder_swiginit(self, _mupdf.new_fz_text_decoder())
    __swig_destroy__ = _mupdf.delete_fz_text_decoder