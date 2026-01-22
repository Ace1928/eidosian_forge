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
class PdfSanitizeFilterOptions2(PdfSanitizeFilterOptions):
    """ Wrapper class for struct pdf_sanitize_filter_options with virtual fns for each fnptr; this is for use as a SWIG Director class."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self):
        """ == Constructor."""
        if self.__class__ == PdfSanitizeFilterOptions2:
            _self = None
        else:
            _self = self
        _mupdf.PdfSanitizeFilterOptions2_swiginit(self, _mupdf.new_PdfSanitizeFilterOptions2(_self))

    def use_virtual_image_filter(self, use=True):
        """
        These methods set the function pointers in *m_internal
        to point to internal callbacks that call our virtual methods.
        """
        return _mupdf.PdfSanitizeFilterOptions2_use_virtual_image_filter(self, use)

    def use_virtual_text_filter(self, use=True):
        return _mupdf.PdfSanitizeFilterOptions2_use_virtual_text_filter(self, use)

    def use_virtual_after_text_object(self, use=True):
        return _mupdf.PdfSanitizeFilterOptions2_use_virtual_after_text_object(self, use)

    def use_virtual_culler(self, use=True):
        return _mupdf.PdfSanitizeFilterOptions2_use_virtual_culler(self, use)

    def image_filter(self, arg_0, arg_2, arg_3, arg_4, arg_5):
        """ Default virtual method implementations; these all throw an exception."""
        return _mupdf.PdfSanitizeFilterOptions2_image_filter(self, arg_0, arg_2, arg_3, arg_4, arg_5)

    def text_filter(self, arg_0, arg_2, arg_3, arg_4, arg_5, arg_6):
        return _mupdf.PdfSanitizeFilterOptions2_text_filter(self, arg_0, arg_2, arg_3, arg_4, arg_5, arg_6)

    def after_text_object(self, arg_0, arg_2, arg_3, arg_4):
        return _mupdf.PdfSanitizeFilterOptions2_after_text_object(self, arg_0, arg_2, arg_3, arg_4)

    def culler(self, arg_0, arg_2, arg_3):
        return _mupdf.PdfSanitizeFilterOptions2_culler(self, arg_0, arg_2, arg_3)
    __swig_destroy__ = _mupdf.delete_PdfSanitizeFilterOptions2

    def __disown__(self):
        self.this.disown()
        _mupdf.disown_PdfSanitizeFilterOptions2(self)
        return weakref.proxy(self)