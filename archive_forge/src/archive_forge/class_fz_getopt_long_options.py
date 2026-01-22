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
class fz_getopt_long_options(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    option = property(_mupdf.fz_getopt_long_options_option_get, _mupdf.fz_getopt_long_options_option_set)
    flag = property(_mupdf.fz_getopt_long_options_flag_get, _mupdf.fz_getopt_long_options_flag_set)
    opaque = property(_mupdf.fz_getopt_long_options_opaque_get, _mupdf.fz_getopt_long_options_opaque_set)

    def __init__(self):
        _mupdf.fz_getopt_long_options_swiginit(self, _mupdf.new_fz_getopt_long_options())
    __swig_destroy__ = _mupdf.delete_fz_getopt_long_options