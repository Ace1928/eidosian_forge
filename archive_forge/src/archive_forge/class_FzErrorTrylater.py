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
class FzErrorTrylater(FzErrorBase):
    """ For `FZ_ERROR_TRYLATER`."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self, message):
        _mupdf.FzErrorTrylater_swiginit(self, _mupdf.new_FzErrorTrylater(message))
    __swig_destroy__ = _mupdf.delete_FzErrorTrylater