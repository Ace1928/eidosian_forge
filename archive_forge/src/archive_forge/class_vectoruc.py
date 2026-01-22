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
class vectoruc(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def iterator(self):
        return _mupdf.vectoruc_iterator(self)

    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _mupdf.vectoruc___nonzero__(self)

    def __bool__(self):
        return _mupdf.vectoruc___bool__(self)

    def __len__(self):
        return _mupdf.vectoruc___len__(self)

    def __getslice__(self, i, j):
        return _mupdf.vectoruc___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _mupdf.vectoruc___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _mupdf.vectoruc___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _mupdf.vectoruc___delitem__(self, *args)

    def __getitem__(self, *args):
        return _mupdf.vectoruc___getitem__(self, *args)

    def __setitem__(self, *args):
        return _mupdf.vectoruc___setitem__(self, *args)

    def pop(self):
        return _mupdf.vectoruc_pop(self)

    def append(self, x):
        return _mupdf.vectoruc_append(self, x)

    def empty(self):
        return _mupdf.vectoruc_empty(self)

    def size(self):
        return _mupdf.vectoruc_size(self)

    def swap(self, v):
        return _mupdf.vectoruc_swap(self, v)

    def begin(self):
        return _mupdf.vectoruc_begin(self)

    def end(self):
        return _mupdf.vectoruc_end(self)

    def rbegin(self):
        return _mupdf.vectoruc_rbegin(self)

    def rend(self):
        return _mupdf.vectoruc_rend(self)

    def clear(self):
        return _mupdf.vectoruc_clear(self)

    def get_allocator(self):
        return _mupdf.vectoruc_get_allocator(self)

    def pop_back(self):
        return _mupdf.vectoruc_pop_back(self)

    def erase(self, *args):
        return _mupdf.vectoruc_erase(self, *args)

    def __init__(self, *args):
        _mupdf.vectoruc_swiginit(self, _mupdf.new_vectoruc(*args))

    def push_back(self, x):
        return _mupdf.vectoruc_push_back(self, x)

    def front(self):
        return _mupdf.vectoruc_front(self)

    def back(self):
        return _mupdf.vectoruc_back(self)

    def assign(self, n, x):
        return _mupdf.vectoruc_assign(self, n, x)

    def resize(self, *args):
        return _mupdf.vectoruc_resize(self, *args)

    def insert(self, *args):
        return _mupdf.vectoruc_insert(self, *args)

    def reserve(self, n):
        return _mupdf.vectoruc_reserve(self, n)

    def capacity(self):
        return _mupdf.vectoruc_capacity(self)
    __swig_destroy__ = _mupdf.delete_vectoruc