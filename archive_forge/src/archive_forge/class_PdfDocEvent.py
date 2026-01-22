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
class PdfDocEvent(object):
    """ Wrapper class for struct `pdf_doc_event`. Not copyable or assignable."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def pdf_access_exec_menu_item_event(self):
        """ Class-aware wrapper for `::pdf_access_exec_menu_item_event()`."""
        return _mupdf.PdfDocEvent_pdf_access_exec_menu_item_event(self)

    def __init__(self, *args):
        """
        *Overload 1:*
        Default constructor, sets `m_internal` to null.

        |

        *Overload 2:*
        Constructor using raw copy of pre-existing `::pdf_doc_event`.
        """
        _mupdf.PdfDocEvent_swiginit(self, _mupdf.new_PdfDocEvent(*args))
    __swig_destroy__ = _mupdf.delete_PdfDocEvent

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.PdfDocEvent_m_internal_value(self)
    m_internal = property(_mupdf.PdfDocEvent_m_internal_get, _mupdf.PdfDocEvent_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.PdfDocEvent_s_num_instances_get, _mupdf.PdfDocEvent_s_num_instances_set)