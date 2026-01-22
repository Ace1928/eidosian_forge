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
class FzArc4(object):
    """
    Wrapper class for struct `fz_arc4`. Not copyable or assignable.
    Structure definition is public to enable stack
    based allocation. Do not access the members directly.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def fz_arc4_encrypt(self, dest, src, len):
        """
        Class-aware wrapper for `::fz_arc4_encrypt()`.
        	RC4 block encrypt operation; encrypt src into dst (both of
        	length len) updating the RC4 state as we go.

        	Never throws an exception.
        """
        return _mupdf.FzArc4_fz_arc4_encrypt(self, dest, src, len)

    def fz_arc4_final(self):
        """
        Class-aware wrapper for `::fz_arc4_final()`.
        	RC4 finalization. Zero the context.

        	Never throws an exception.
        """
        return _mupdf.FzArc4_fz_arc4_final(self)

    def fz_arc4_init(self, key, len):
        """
        Class-aware wrapper for `::fz_arc4_init()`.
        	RC4 initialization. Begins an RC4 operation, writing a new
        	context.

        	Never throws an exception.
        """
        return _mupdf.FzArc4_fz_arc4_init(self, key, len)

    def __init__(self, *args):
        """
        *Overload 1:*
        Default constructor, sets `m_internal` to null.

        |

        *Overload 2:*
        Constructor using raw copy of pre-existing `::fz_arc4`.
        """
        _mupdf.FzArc4_swiginit(self, _mupdf.new_FzArc4(*args))
    __swig_destroy__ = _mupdf.delete_FzArc4

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzArc4_m_internal_value(self)
    m_internal = property(_mupdf.FzArc4_m_internal_get, _mupdf.FzArc4_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzArc4_s_num_instances_get, _mupdf.FzArc4_s_num_instances_set)