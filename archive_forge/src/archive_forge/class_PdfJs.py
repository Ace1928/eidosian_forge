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
class PdfJs(object):
    """ Wrapper class for struct `pdf_js`. Not copyable or assignable."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def pdf_js_event_init(self, target, value, willCommit):
        """ Class-aware wrapper for `::pdf_js_event_init()`."""
        return _mupdf.PdfJs_pdf_js_event_init(self, target, value, willCommit)

    def pdf_js_event_init_keystroke(self, target, evt):
        """ Class-aware wrapper for `::pdf_js_event_init_keystroke()`."""
        return _mupdf.PdfJs_pdf_js_event_init_keystroke(self, target, evt)

    def pdf_js_event_result(self):
        """ Class-aware wrapper for `::pdf_js_event_result()`."""
        return _mupdf.PdfJs_pdf_js_event_result(self)

    def pdf_js_event_result_keystroke(self, evt):
        """ Class-aware wrapper for `::pdf_js_event_result_keystroke()`."""
        return _mupdf.PdfJs_pdf_js_event_result_keystroke(self, evt)

    def pdf_js_event_result_validate(self, newvalue):
        """
        Class-aware wrapper for `::pdf_js_event_result_validate()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_js_event_result_validate()` => `(int, char *newvalue)`
        """
        return _mupdf.PdfJs_pdf_js_event_result_validate(self, newvalue)

    def pdf_js_event_value(self):
        """ Class-aware wrapper for `::pdf_js_event_value()`."""
        return _mupdf.PdfJs_pdf_js_event_value(self)

    def pdf_js_execute(self, name, code, result):
        """
        Class-aware wrapper for `::pdf_js_execute()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_js_execute(const char *name, const char *code)` => char *result
        """
        return _mupdf.PdfJs_pdf_js_execute(self, name, code, result)

    def __init__(self, *args):
        """
        *Overload 1:*
        Default constructor, sets `m_internal` to null.

        |

        *Overload 2:*
        Constructor using raw copy of pre-existing `::pdf_js`.
        """
        _mupdf.PdfJs_swiginit(self, _mupdf.new_PdfJs(*args))
    __swig_destroy__ = _mupdf.delete_PdfJs

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.PdfJs_m_internal_value(self)
    m_internal = property(_mupdf.PdfJs_m_internal_get, _mupdf.PdfJs_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.PdfJs_s_num_instances_get, _mupdf.PdfJs_s_num_instances_set)