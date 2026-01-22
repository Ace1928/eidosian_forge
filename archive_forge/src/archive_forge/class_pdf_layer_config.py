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
class pdf_layer_config(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    name = property(_mupdf.pdf_layer_config_name_get, _mupdf.pdf_layer_config_name_set)
    creator = property(_mupdf.pdf_layer_config_creator_get, _mupdf.pdf_layer_config_creator_set)

    def __init__(self):
        _mupdf.pdf_layer_config_swiginit(self, _mupdf.new_pdf_layer_config())
    __swig_destroy__ = _mupdf.delete_pdf_layer_config