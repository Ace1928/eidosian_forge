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
class pdf_xref_subsec(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    next = property(_mupdf.pdf_xref_subsec_next_get, _mupdf.pdf_xref_subsec_next_set)
    len = property(_mupdf.pdf_xref_subsec_len_get, _mupdf.pdf_xref_subsec_len_set)
    start = property(_mupdf.pdf_xref_subsec_start_get, _mupdf.pdf_xref_subsec_start_set)
    table = property(_mupdf.pdf_xref_subsec_table_get, _mupdf.pdf_xref_subsec_table_set)

    def __init__(self):
        _mupdf.pdf_xref_subsec_swiginit(self, _mupdf.new_pdf_xref_subsec())
    __swig_destroy__ = _mupdf.delete_pdf_xref_subsec