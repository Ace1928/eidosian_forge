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
class fz_compressed_buffer(object):
    """
    Buffers of compressed data; typically for the source data
    for images.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    refs = property(_mupdf.fz_compressed_buffer_refs_get, _mupdf.fz_compressed_buffer_refs_set)
    params = property(_mupdf.fz_compressed_buffer_params_get, _mupdf.fz_compressed_buffer_params_set)
    buffer = property(_mupdf.fz_compressed_buffer_buffer_get, _mupdf.fz_compressed_buffer_buffer_set)

    def __init__(self):
        _mupdf.fz_compressed_buffer_swiginit(self, _mupdf.new_fz_compressed_buffer())
    __swig_destroy__ = _mupdf.delete_fz_compressed_buffer