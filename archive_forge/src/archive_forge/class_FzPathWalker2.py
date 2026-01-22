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
class FzPathWalker2(FzPathWalker):
    """ Wrapper class for struct fz_path_walker with virtual fns for each fnptr; this is for use as a SWIG Director class."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self):
        """ == Constructor."""
        if self.__class__ == FzPathWalker2:
            _self = None
        else:
            _self = self
        _mupdf.FzPathWalker2_swiginit(self, _mupdf.new_FzPathWalker2(_self))
    __swig_destroy__ = _mupdf.delete_FzPathWalker2

    def use_virtual_moveto(self, use=True):
        """
        These methods set the function pointers in *m_internal
        to point to internal callbacks that call our virtual methods.
        """
        return _mupdf.FzPathWalker2_use_virtual_moveto(self, use)

    def use_virtual_lineto(self, use=True):
        return _mupdf.FzPathWalker2_use_virtual_lineto(self, use)

    def use_virtual_curveto(self, use=True):
        return _mupdf.FzPathWalker2_use_virtual_curveto(self, use)

    def use_virtual_closepath(self, use=True):
        return _mupdf.FzPathWalker2_use_virtual_closepath(self, use)

    def use_virtual_quadto(self, use=True):
        return _mupdf.FzPathWalker2_use_virtual_quadto(self, use)

    def use_virtual_curvetov(self, use=True):
        return _mupdf.FzPathWalker2_use_virtual_curvetov(self, use)

    def use_virtual_curvetoy(self, use=True):
        return _mupdf.FzPathWalker2_use_virtual_curvetoy(self, use)

    def use_virtual_rectto(self, use=True):
        return _mupdf.FzPathWalker2_use_virtual_rectto(self, use)

    def moveto(self, arg_0, arg_2, arg_3):
        """ Default virtual method implementations; these all throw an exception."""
        return _mupdf.FzPathWalker2_moveto(self, arg_0, arg_2, arg_3)

    def lineto(self, arg_0, arg_2, arg_3):
        return _mupdf.FzPathWalker2_lineto(self, arg_0, arg_2, arg_3)

    def curveto(self, arg_0, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7):
        return _mupdf.FzPathWalker2_curveto(self, arg_0, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7)

    def closepath(self, arg_0):
        return _mupdf.FzPathWalker2_closepath(self, arg_0)

    def quadto(self, arg_0, arg_2, arg_3, arg_4, arg_5):
        return _mupdf.FzPathWalker2_quadto(self, arg_0, arg_2, arg_3, arg_4, arg_5)

    def curvetov(self, arg_0, arg_2, arg_3, arg_4, arg_5):
        return _mupdf.FzPathWalker2_curvetov(self, arg_0, arg_2, arg_3, arg_4, arg_5)

    def curvetoy(self, arg_0, arg_2, arg_3, arg_4, arg_5):
        return _mupdf.FzPathWalker2_curvetoy(self, arg_0, arg_2, arg_3, arg_4, arg_5)

    def rectto(self, arg_0, arg_2, arg_3, arg_4, arg_5):
        return _mupdf.FzPathWalker2_rectto(self, arg_0, arg_2, arg_3, arg_4, arg_5)

    def __disown__(self):
        self.this.disown()
        _mupdf.disown_FzPathWalker2(self)
        return weakref.proxy(self)