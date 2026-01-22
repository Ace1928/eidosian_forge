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
class FzOutput2(FzOutput):
    """ Wrapper class for struct fz_output with virtual fns for each fnptr; this is for use as a SWIG Director class."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self):
        """ == Constructor."""
        if self.__class__ == FzOutput2:
            _self = None
        else:
            _self = self
        _mupdf.FzOutput2_swiginit(self, _mupdf.new_FzOutput2(_self))

    def use_virtual_write(self, use=True):
        """
        These methods set the function pointers in *m_internal
        to point to internal callbacks that call our virtual methods.
        """
        return _mupdf.FzOutput2_use_virtual_write(self, use)

    def use_virtual_seek(self, use=True):
        return _mupdf.FzOutput2_use_virtual_seek(self, use)

    def use_virtual_tell(self, use=True):
        return _mupdf.FzOutput2_use_virtual_tell(self, use)

    def use_virtual_close(self, use=True):
        return _mupdf.FzOutput2_use_virtual_close(self, use)

    def use_virtual_drop(self, use=True):
        return _mupdf.FzOutput2_use_virtual_drop(self, use)

    def use_virtual_as_stream(self, use=True):
        return _mupdf.FzOutput2_use_virtual_as_stream(self, use)

    def use_virtual_truncate(self, use=True):
        return _mupdf.FzOutput2_use_virtual_truncate(self, use)

    def write(self, arg_0, arg_2, arg_3):
        """ Default virtual method implementations; these all throw an exception."""
        return _mupdf.FzOutput2_write(self, arg_0, arg_2, arg_3)

    def seek(self, arg_0, arg_2, arg_3):
        return _mupdf.FzOutput2_seek(self, arg_0, arg_2, arg_3)

    def tell(self, arg_0):
        return _mupdf.FzOutput2_tell(self, arg_0)

    def close(self, arg_0):
        return _mupdf.FzOutput2_close(self, arg_0)

    def drop(self, arg_0):
        return _mupdf.FzOutput2_drop(self, arg_0)

    def as_stream(self, arg_0):
        return _mupdf.FzOutput2_as_stream(self, arg_0)

    def truncate(self, arg_0):
        return _mupdf.FzOutput2_truncate(self, arg_0)
    __swig_destroy__ = _mupdf.delete_FzOutput2

    def __disown__(self):
        self.this.disown()
        _mupdf.disown_FzOutput2(self)
        return weakref.proxy(self)