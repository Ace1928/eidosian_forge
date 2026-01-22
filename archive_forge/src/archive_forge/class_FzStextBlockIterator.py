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
class FzStextBlockIterator(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self, item):
        _mupdf.FzStextBlockIterator_swiginit(self, _mupdf.new_FzStextBlockIterator(item))

    def __increment__(self):
        return _mupdf.FzStextBlockIterator___increment__(self)

    def __eq__(self, rhs):
        return _mupdf.FzStextBlockIterator___eq__(self, rhs)

    def __ne__(self, rhs):
        return _mupdf.FzStextBlockIterator___ne__(self, rhs)

    def __ref__(self):
        return _mupdf.FzStextBlockIterator___ref__(self)

    def __deref__(self):
        return _mupdf.FzStextBlockIterator___deref__(self)
    __swig_destroy__ = _mupdf.delete_FzStextBlockIterator

    def begin(self):
        """ Used for iteration over linked list of FzStextChar items starting at fz_stext_char::first_char."""
        return _mupdf.FzStextBlockIterator_begin(self)

    def end(self):
        """ Used for iteration over linked list of FzStextChar items starting at fz_stext_char::first_char."""
        return _mupdf.FzStextBlockIterator_end(self)

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzStextBlockIterator_m_internal_value(self)
    m_internal = property(_mupdf.FzStextBlockIterator_m_internal_get, _mupdf.FzStextBlockIterator_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzStextBlockIterator_s_num_instances_get, _mupdf.FzStextBlockIterator_s_num_instances_set)