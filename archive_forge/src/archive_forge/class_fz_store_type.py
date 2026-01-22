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
class fz_store_type(object):
    """
    Every type of object to be placed into the store defines an
    fz_store_type. This contains the pointers to functions to
    make hashes, manipulate keys, and check for needing reaping.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    name = property(_mupdf.fz_store_type_name_get, _mupdf.fz_store_type_name_set)
    make_hash_key = property(_mupdf.fz_store_type_make_hash_key_get, _mupdf.fz_store_type_make_hash_key_set)
    keep_key = property(_mupdf.fz_store_type_keep_key_get, _mupdf.fz_store_type_keep_key_set)
    drop_key = property(_mupdf.fz_store_type_drop_key_get, _mupdf.fz_store_type_drop_key_set)
    cmp_key = property(_mupdf.fz_store_type_cmp_key_get, _mupdf.fz_store_type_cmp_key_set)
    format_key = property(_mupdf.fz_store_type_format_key_get, _mupdf.fz_store_type_format_key_set)
    needs_reap = property(_mupdf.fz_store_type_needs_reap_get, _mupdf.fz_store_type_needs_reap_set)

    def __init__(self):
        _mupdf.fz_store_type_swiginit(self, _mupdf.new_fz_store_type())
    __swig_destroy__ = _mupdf.delete_fz_store_type