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
class FzKeyStorable(object):
    """
    Wrapper class for struct `fz_key_storable`.
    Any storable object that can appear in the key of another
    storable object should include an fz_key_storable structure
    at the start (by convention at least) of their structure.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self, *args):
        """
        *Overload 1:*
        Copy constructor using `fz_keep_key_storable()`.

        |

        *Overload 2:*
        Default constructor, sets `m_internal` to null.

        |

        *Overload 3:*
        Constructor using raw copy of pre-existing `::fz_key_storable`.
        """
        _mupdf.FzKeyStorable_swiginit(self, _mupdf.new_FzKeyStorable(*args))
    __swig_destroy__ = _mupdf.delete_FzKeyStorable

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzKeyStorable_m_internal_value(self)
    m_internal = property(_mupdf.FzKeyStorable_m_internal_get, _mupdf.FzKeyStorable_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzKeyStorable_s_num_instances_get, _mupdf.FzKeyStorable_s_num_instances_set)