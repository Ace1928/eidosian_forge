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
class fz_warn_context(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    print_user = property(_mupdf.fz_warn_context_print_user_get, _mupdf.fz_warn_context_print_user_set)
    _print = property(_mupdf.fz_warn_context__print_get, _mupdf.fz_warn_context__print_set)
    count = property(_mupdf.fz_warn_context_count_get, _mupdf.fz_warn_context_count_set)
    message = property(_mupdf.fz_warn_context_message_get, _mupdf.fz_warn_context_message_set)

    def __init__(self):
        _mupdf.fz_warn_context_swiginit(self, _mupdf.new_fz_warn_context())
    __swig_destroy__ = _mupdf.delete_fz_warn_context