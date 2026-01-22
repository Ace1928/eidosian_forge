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
class FzStextLine(object):
    """ Wrapper class for struct `fz_stext_line`."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def begin(self):
        """ Used for iteration over linked list of FzStextChar items starting at fz_stext_char::first_char."""
        return _mupdf.FzStextLine_begin(self)

    def end(self):
        """ Used for iteration over linked list of FzStextChar items starting at fz_stext_char::first_char."""
        return _mupdf.FzStextLine_end(self)

    def __init__(self, *args):
        """
        *Overload 1:*
        We use default copy constructor and operator=.  Default constructor, sets `m_internal` to null.

        |

        *Overload 2:*
        Constructor using raw copy of pre-existing `::fz_stext_line`.
        """
        _mupdf.FzStextLine_swiginit(self, _mupdf.new_FzStextLine(*args))
    __swig_destroy__ = _mupdf.delete_FzStextLine

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzStextLine_m_internal_value(self)
    m_internal = property(_mupdf.FzStextLine_m_internal_get, _mupdf.FzStextLine_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzStextLine_s_num_instances_get, _mupdf.FzStextLine_s_num_instances_set)