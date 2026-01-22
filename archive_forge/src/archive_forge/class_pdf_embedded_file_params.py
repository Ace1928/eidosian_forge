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
class pdf_embedded_file_params(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    filename = property(_mupdf.pdf_embedded_file_params_filename_get, _mupdf.pdf_embedded_file_params_filename_set)
    mimetype = property(_mupdf.pdf_embedded_file_params_mimetype_get, _mupdf.pdf_embedded_file_params_mimetype_set)
    size = property(_mupdf.pdf_embedded_file_params_size_get, _mupdf.pdf_embedded_file_params_size_set)
    created = property(_mupdf.pdf_embedded_file_params_created_get, _mupdf.pdf_embedded_file_params_created_set)
    modified = property(_mupdf.pdf_embedded_file_params_modified_get, _mupdf.pdf_embedded_file_params_modified_set)

    def __init__(self):
        _mupdf.pdf_embedded_file_params_swiginit(self, _mupdf.new_pdf_embedded_file_params())
    __swig_destroy__ = _mupdf.delete_pdf_embedded_file_params