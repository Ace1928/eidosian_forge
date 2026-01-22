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
class PdfFunction(object):
    """ Wrapper class for struct `pdf_function`."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def pdf_eval_function(self, _in, inlen, out, outlen):
        """
        Class-aware wrapper for `::pdf_eval_function()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_eval_function(const float *in, int inlen, int outlen)` => float out
        """
        return _mupdf.PdfFunction_pdf_eval_function(self, _in, inlen, out, outlen)

    def pdf_function_size(self):
        """ Class-aware wrapper for `::pdf_function_size()`."""
        return _mupdf.PdfFunction_pdf_function_size(self)

    def __init__(self, *args):
        """
        *Overload 1:*
        Copy constructor using `pdf_keep_function()`.

        |

        *Overload 2:*
        Default constructor, sets `m_internal` to null.

        |

        *Overload 3:*
        Constructor using raw copy of pre-existing `::pdf_function`.
        """
        _mupdf.PdfFunction_swiginit(self, _mupdf.new_PdfFunction(*args))
    __swig_destroy__ = _mupdf.delete_PdfFunction

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.PdfFunction_m_internal_value(self)
    m_internal = property(_mupdf.PdfFunction_m_internal_get, _mupdf.PdfFunction_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.PdfFunction_s_num_instances_get, _mupdf.PdfFunction_s_num_instances_set)