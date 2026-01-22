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
class FzMd5(object):
    """
    Wrapper class for struct `fz_md5`.
    Structure definition is public to enable stack
    based allocation. Do not access the members directly.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def fz_md5_final(self, digest):
        """
        We use default copy constructor and operator=.  Class-aware wrapper for `::fz_md5_final()`.
        	MD5 finalization. Ends an MD5 message-digest operation, writing
        	the message digest and zeroizing the context.

        	Never throws an exception.
        """
        return _mupdf.FzMd5_fz_md5_final(self, digest)

    def fz_md5_final2(self):
        """
        Class-aware wrapper for `::fz_md5_final2()`.
        C++ alternative to fz_md5_final() that returns the digest by value.
        """
        return _mupdf.FzMd5_fz_md5_final2(self)

    def fz_md5_init(self):
        """
        Class-aware wrapper for `::fz_md5_init()`.
        	MD5 initialization. Begins an MD5 operation, writing a new
        	context.

        	Never throws an exception.
        """
        return _mupdf.FzMd5_fz_md5_init(self)

    def fz_md5_update(self, input, inlen):
        """
        Class-aware wrapper for `::fz_md5_update()`.
        	MD5 block update operation. Continues an MD5 message-digest
        	operation, processing another message block, and updating the
        	context.

        	Never throws an exception.
        """
        return _mupdf.FzMd5_fz_md5_update(self, input, inlen)

    def fz_md5_update_int64(self, i):
        """
        Class-aware wrapper for `::fz_md5_update_int64()`.
        	MD5 block update operation. Continues an MD5 message-digest
        	operation, processing an int64, and updating the context.

        	Never throws an exception.
        """
        return _mupdf.FzMd5_fz_md5_update_int64(self, i)

    def __init__(self, *args):
        """
        *Overload 1:*
        Default constructor calls md5_init().

        |

        *Overload 2:*
        Constructor using raw copy of pre-existing `::fz_md5`.
        """
        _mupdf.FzMd5_swiginit(self, _mupdf.new_FzMd5(*args))

    def lo(self):
        return _mupdf.FzMd5_lo(self)

    def hi(self):
        return _mupdf.FzMd5_hi(self)

    def a(self):
        return _mupdf.FzMd5_a(self)

    def b(self):
        return _mupdf.FzMd5_b(self)

    def c(self):
        return _mupdf.FzMd5_c(self)

    def d(self):
        return _mupdf.FzMd5_d(self)

    def buffer(self):
        return _mupdf.FzMd5_buffer(self)
    __swig_destroy__ = _mupdf.delete_FzMd5
    m_internal = property(_mupdf.FzMd5_m_internal_get, _mupdf.FzMd5_m_internal_set)
    s_num_instances = property(_mupdf.FzMd5_s_num_instances_get, _mupdf.FzMd5_s_num_instances_set, doc=' Wrapped data is held by value.')

    def to_string(self):
        """ Returns string containing our members, labelled and inside (...), using operator<<."""
        return _mupdf.FzMd5_to_string(self)

    def __eq__(self, rhs):
        """ Comparison method."""
        return _mupdf.FzMd5___eq__(self, rhs)

    def __ne__(self, rhs):
        """ Comparison method."""
        return _mupdf.FzMd5___ne__(self, rhs)