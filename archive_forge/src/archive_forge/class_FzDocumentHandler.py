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
class FzDocumentHandler(object):
    """ Wrapper class for struct `fz_document_handler`. Not copyable or assignable."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def fz_register_document_handler(self):
        """
        Class-aware wrapper for `::fz_register_document_handler()`.
        	Register a handler for a document type.

        	handler: The handler to register.
        """
        return _mupdf.FzDocumentHandler_fz_register_document_handler(self)

    def __init__(self, *args):
        """
        *Overload 1:*
        Default constructor, sets `m_internal` to null.

        |

        *Overload 2:*
        Constructor using raw copy of pre-existing `::fz_document_handler`.
        """
        _mupdf.FzDocumentHandler_swiginit(self, _mupdf.new_FzDocumentHandler(*args))
    __swig_destroy__ = _mupdf.delete_FzDocumentHandler

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzDocumentHandler_m_internal_value(self)
    m_internal = property(_mupdf.FzDocumentHandler_m_internal_get, _mupdf.FzDocumentHandler_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzDocumentHandler_s_num_instances_get, _mupdf.FzDocumentHandler_s_num_instances_set)