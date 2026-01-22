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
class fz_storable(object):
    """
    Any storable object should include an fz_storable structure
    at the start (by convention at least) of their structure.
    (Unless it starts with an fz_key_storable, see below).
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    refs = property(_mupdf.fz_storable_refs_get, _mupdf.fz_storable_refs_set)
    drop = property(_mupdf.fz_storable_drop_get, _mupdf.fz_storable_drop_set)
    droppable = property(_mupdf.fz_storable_droppable_get, _mupdf.fz_storable_droppable_set)

    def __init__(self):
        _mupdf.fz_storable_swiginit(self, _mupdf.new_fz_storable())
    __swig_destroy__ = _mupdf.delete_fz_storable