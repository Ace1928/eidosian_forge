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
class PdfGraftMap(object):
    """ Wrapper class for struct `pdf_graft_map`."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def pdf_graft_mapped_object(self, obj):
        """ Class-aware wrapper for `::pdf_graft_mapped_object()`."""
        return _mupdf.PdfGraftMap_pdf_graft_mapped_object(self, obj)

    def pdf_graft_mapped_page(self, page_to, src, page_from):
        """ Class-aware wrapper for `::pdf_graft_mapped_page()`."""
        return _mupdf.PdfGraftMap_pdf_graft_mapped_page(self, page_to, src, page_from)

    def __init__(self, *args):
        """
        *Overload 1:*
        == Constructors.  Constructor using `pdf_new_graft_map()`.

        |

        *Overload 2:*
        Copy constructor using `pdf_keep_graft_map()`.

        |

        *Overload 3:*
        Default constructor, sets `m_internal` to null.

        |

        *Overload 4:*
        Constructor using raw copy of pre-existing `::pdf_graft_map`.
        """
        _mupdf.PdfGraftMap_swiginit(self, _mupdf.new_PdfGraftMap(*args))
    __swig_destroy__ = _mupdf.delete_PdfGraftMap

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.PdfGraftMap_m_internal_value(self)
    m_internal = property(_mupdf.PdfGraftMap_m_internal_get, _mupdf.PdfGraftMap_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.PdfGraftMap_s_num_instances_get, _mupdf.PdfGraftMap_s_num_instances_set)