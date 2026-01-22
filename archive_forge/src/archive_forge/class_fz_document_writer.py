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
class fz_document_writer(object):
    """
    Structure is public to allow other structures to
    be derived from it. Do not access members directly.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    begin_page = property(_mupdf.fz_document_writer_begin_page_get, _mupdf.fz_document_writer_begin_page_set)
    end_page = property(_mupdf.fz_document_writer_end_page_get, _mupdf.fz_document_writer_end_page_set)
    close_writer = property(_mupdf.fz_document_writer_close_writer_get, _mupdf.fz_document_writer_close_writer_set)
    drop_writer = property(_mupdf.fz_document_writer_drop_writer_get, _mupdf.fz_document_writer_drop_writer_set)
    dev = property(_mupdf.fz_document_writer_dev_get, _mupdf.fz_document_writer_dev_set)

    def __init__(self):
        _mupdf.fz_document_writer_swiginit(self, _mupdf.new_fz_document_writer())
    __swig_destroy__ = _mupdf.delete_fz_document_writer