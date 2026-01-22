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
class FzString(object):
    """ Wrapper class for struct `fz_string`."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self, *args):
        """
        *Overload 1:*
        == Constructors.  Constructor using `fz_new_string()`.

        |

        *Overload 2:*
        Copy constructor using `fz_keep_string()`.

        |

        *Overload 3:*
        Default constructor, sets `m_internal` to null.

        |

        *Overload 4:*
        Constructor using raw copy of pre-existing `::fz_string`.
        """
        _mupdf.FzString_swiginit(self, _mupdf.new_FzString(*args))
    __swig_destroy__ = _mupdf.delete_FzString

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzString_m_internal_value(self)
    m_internal = property(_mupdf.FzString_m_internal_get, _mupdf.FzString_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzString_s_num_instances_get, _mupdf.FzString_s_num_instances_set)