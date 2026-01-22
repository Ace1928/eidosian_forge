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
class fz_document_handler(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    recognize = property(_mupdf.fz_document_handler_recognize_get, _mupdf.fz_document_handler_recognize_set)
    open = property(_mupdf.fz_document_handler_open_get, _mupdf.fz_document_handler_open_set)
    extensions = property(_mupdf.fz_document_handler_extensions_get, _mupdf.fz_document_handler_extensions_set)
    mimetypes = property(_mupdf.fz_document_handler_mimetypes_get, _mupdf.fz_document_handler_mimetypes_set)
    recognize_content = property(_mupdf.fz_document_handler_recognize_content_get, _mupdf.fz_document_handler_recognize_content_set)

    def __init__(self):
        _mupdf.fz_document_handler_swiginit(self, _mupdf.new_fz_document_handler())
    __swig_destroy__ = _mupdf.delete_fz_document_handler