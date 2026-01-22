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
class PdfTextObjectState(object):
    """ Wrapper class for struct `pdf_text_object_state`. Not copyable or assignable."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def pdf_tos_get_text(self):
        """ Class-aware wrapper for `::pdf_tos_get_text()`."""
        return _mupdf.PdfTextObjectState_pdf_tos_get_text(self)

    def pdf_tos_make_trm(self, text, fontdesc, cid, trm):
        """ Class-aware wrapper for `::pdf_tos_make_trm()`."""
        return _mupdf.PdfTextObjectState_pdf_tos_make_trm(self, text, fontdesc, cid, trm)

    def pdf_tos_move_after_char(self):
        """ Class-aware wrapper for `::pdf_tos_move_after_char()`."""
        return _mupdf.PdfTextObjectState_pdf_tos_move_after_char(self)

    def pdf_tos_newline(self, leading):
        """ Class-aware wrapper for `::pdf_tos_newline()`."""
        return _mupdf.PdfTextObjectState_pdf_tos_newline(self, leading)

    def pdf_tos_reset(self, render):
        """ Class-aware wrapper for `::pdf_tos_reset()`."""
        return _mupdf.PdfTextObjectState_pdf_tos_reset(self, render)

    def pdf_tos_set_matrix(self, a, b, c, d, e, f):
        """ Class-aware wrapper for `::pdf_tos_set_matrix()`."""
        return _mupdf.PdfTextObjectState_pdf_tos_set_matrix(self, a, b, c, d, e, f)

    def pdf_tos_translate(self, tx, ty):
        """ Class-aware wrapper for `::pdf_tos_translate()`."""
        return _mupdf.PdfTextObjectState_pdf_tos_translate(self, tx, ty)

    def __init__(self, *args):
        """
        *Overload 1:*
        Default constructor, sets `m_internal` to null.

        |

        *Overload 2:*
        Constructor using raw copy of pre-existing `::pdf_text_object_state`.
        """
        _mupdf.PdfTextObjectState_swiginit(self, _mupdf.new_PdfTextObjectState(*args))
    __swig_destroy__ = _mupdf.delete_PdfTextObjectState

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.PdfTextObjectState_m_internal_value(self)
    m_internal = property(_mupdf.PdfTextObjectState_m_internal_get, _mupdf.PdfTextObjectState_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.PdfTextObjectState_s_num_instances_get, _mupdf.PdfTextObjectState_s_num_instances_set)