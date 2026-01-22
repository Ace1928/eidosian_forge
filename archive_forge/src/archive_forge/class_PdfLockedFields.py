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
class PdfLockedFields(object):
    """ Wrapper class for struct `pdf_locked_fields`. Not copyable or assignable."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def pdf_is_field_locked(self, name):
        """ Class-aware wrapper for `::pdf_is_field_locked()`."""
        return _mupdf.PdfLockedFields_pdf_is_field_locked(self, name)

    def __init__(self, *args):
        """
        *Overload 1:*
        Default constructor, sets `m_internal` to null.

        |

        *Overload 2:*
        Constructor using raw copy of pre-existing `::pdf_locked_fields`.
        """
        _mupdf.PdfLockedFields_swiginit(self, _mupdf.new_PdfLockedFields(*args))
    __swig_destroy__ = _mupdf.delete_PdfLockedFields

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.PdfLockedFields_m_internal_value(self)
    m_internal = property(_mupdf.PdfLockedFields_m_internal_get, _mupdf.PdfLockedFields_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.PdfLockedFields_s_num_instances_get, _mupdf.PdfLockedFields_s_num_instances_set)