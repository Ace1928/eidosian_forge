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
class FzStextLineIterator(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self, item):
        _mupdf.FzStextLineIterator_swiginit(self, _mupdf.new_FzStextLineIterator(item))

    def __increment__(self):
        return _mupdf.FzStextLineIterator___increment__(self)

    def __eq__(self, rhs):
        return _mupdf.FzStextLineIterator___eq__(self, rhs)

    def __ne__(self, rhs):
        return _mupdf.FzStextLineIterator___ne__(self, rhs)

    def __ref__(self):
        return _mupdf.FzStextLineIterator___ref__(self)

    def __deref__(self):
        return _mupdf.FzStextLineIterator___deref__(self)
    __swig_destroy__ = _mupdf.delete_FzStextLineIterator

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzStextLineIterator_m_internal_value(self)
    m_internal = property(_mupdf.FzStextLineIterator_m_internal_get, _mupdf.FzStextLineIterator_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzStextLineIterator_s_num_instances_get, _mupdf.FzStextLineIterator_s_num_instances_set)