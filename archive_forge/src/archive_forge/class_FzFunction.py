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
class FzFunction(object):
    """ Wrapper class for struct `fz_function`."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def fz_eval_function(self, _in, inlen, out, outlen):
        """
        Class-aware wrapper for `::fz_eval_function()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_eval_function(const float *in, int inlen, int outlen)` => float out
        """
        return _mupdf.FzFunction_fz_eval_function(self, _in, inlen, out, outlen)

    def fz_function_size(self):
        """ Class-aware wrapper for `::fz_function_size()`."""
        return _mupdf.FzFunction_fz_function_size(self)

    def __init__(self, *args):
        """
        *Overload 1:*
        == Constructors.  Constructor using `fz_new_function_of_size()`.

        |

        *Overload 2:*
        Copy constructor using `fz_keep_function()`.

        |

        *Overload 3:*
        Default constructor, sets `m_internal` to null.

        |

        *Overload 4:*
        Constructor using raw copy of pre-existing `::fz_function`.
        """
        _mupdf.FzFunction_swiginit(self, _mupdf.new_FzFunction(*args))
    __swig_destroy__ = _mupdf.delete_FzFunction

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzFunction_m_internal_value(self)
    m_internal = property(_mupdf.FzFunction_m_internal_get, _mupdf.FzFunction_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzFunction_s_num_instances_get, _mupdf.FzFunction_s_num_instances_set)