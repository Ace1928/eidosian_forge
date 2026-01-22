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
class PdfLexbuf(object):
    """ Wrapper class for struct `pdf_lexbuf`. Not copyable or assignable."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def pdf_lexbuf_fin(self):
        """ Class-aware wrapper for `::pdf_lexbuf_fin()`."""
        return _mupdf.PdfLexbuf_pdf_lexbuf_fin(self)

    def pdf_lexbuf_grow(self):
        """ Class-aware wrapper for `::pdf_lexbuf_grow()`."""
        return _mupdf.PdfLexbuf_pdf_lexbuf_grow(self)

    def pdf_lexbuf_init(self, size):
        """ Class-aware wrapper for `::pdf_lexbuf_init()`."""
        return _mupdf.PdfLexbuf_pdf_lexbuf_init(self, size)
    __swig_destroy__ = _mupdf.delete_PdfLexbuf

    def __init__(self, *args):
        """
        *Overload 1:*
        Constructor that calls pdf_lexbuf_init(size).

        |

        *Overload 2:*
        Default constructor, sets `m_internal` to null.

        |

        *Overload 3:*
        Constructor using raw copy of pre-existing `::pdf_lexbuf`.
        """
        _mupdf.PdfLexbuf_swiginit(self, _mupdf.new_PdfLexbuf(*args))

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.PdfLexbuf_m_internal_value(self)
    m_internal = property(_mupdf.PdfLexbuf_m_internal_get, _mupdf.PdfLexbuf_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.PdfLexbuf_s_num_instances_get, _mupdf.PdfLexbuf_s_num_instances_set)