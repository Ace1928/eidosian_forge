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
class FzSha256(object):
    """
    Wrapper class for struct `fz_sha256`. Not copyable or assignable.
    Structure definition is public to enable stack
    based allocation. Do not access the members directly.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def fz_sha256_final(self, digest):
        """
        Class-aware wrapper for `::fz_sha256_final()`.
        	MD5 finalization. Ends an MD5 message-digest operation, writing
        	the message digest and zeroizing the context.

        	Never throws an exception.
        """
        return _mupdf.FzSha256_fz_sha256_final(self, digest)

    def fz_sha256_init(self):
        """
        Class-aware wrapper for `::fz_sha256_init()`.
        	SHA256 initialization. Begins an SHA256 operation, initialising
        	the supplied context.

        	Never throws an exception.
        """
        return _mupdf.FzSha256_fz_sha256_init(self)

    def fz_sha256_update(self, input, inlen):
        """
        Class-aware wrapper for `::fz_sha256_update()`.
        	SHA256 block update operation. Continues an SHA256 message-
        	digest operation, processing another message block, and updating
        	the context.

        	Never throws an exception.
        """
        return _mupdf.FzSha256_fz_sha256_update(self, input, inlen)

    def __init__(self, *args):
        """
        *Overload 1:*
        Default constructor, sets `m_internal` to null.

        |

        *Overload 2:*
        Constructor using raw copy of pre-existing `::fz_sha256`.
        """
        _mupdf.FzSha256_swiginit(self, _mupdf.new_FzSha256(*args))
    __swig_destroy__ = _mupdf.delete_FzSha256

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzSha256_m_internal_value(self)
    m_internal = property(_mupdf.FzSha256_m_internal_get, _mupdf.FzSha256_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzSha256_s_num_instances_get, _mupdf.FzSha256_s_num_instances_set)