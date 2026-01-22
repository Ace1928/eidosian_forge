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
class FzLinkDest(object):
    """ Wrapper class for struct `fz_link_dest`. Not copyable or assignable."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def pdf_new_uri_from_explicit_dest(self):
        """ Class-aware wrapper for `::pdf_new_uri_from_explicit_dest()`."""
        return _mupdf.FzLinkDest_pdf_new_uri_from_explicit_dest(self)

    def __init__(self, *args):
        """
        *Overload 1:*
        Default constructor, sets `m_internal` to null.

        |

        *Overload 2:*
        Constructor using raw copy of pre-existing `::fz_link_dest`.
        """
        _mupdf.FzLinkDest_swiginit(self, _mupdf.new_FzLinkDest(*args))
    __swig_destroy__ = _mupdf.delete_FzLinkDest

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzLinkDest_m_internal_value(self)
    m_internal = property(_mupdf.FzLinkDest_m_internal_get, _mupdf.FzLinkDest_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzLinkDest_s_num_instances_get, _mupdf.FzLinkDest_s_num_instances_set)