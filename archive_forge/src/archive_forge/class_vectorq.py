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
class vectorq(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def iterator(self):
        return _mupdf.vectorq_iterator(self)

    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _mupdf.vectorq___nonzero__(self)

    def __bool__(self):
        return _mupdf.vectorq___bool__(self)

    def __len__(self):
        return _mupdf.vectorq___len__(self)

    def __getslice__(self, i, j):
        return _mupdf.vectorq___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _mupdf.vectorq___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _mupdf.vectorq___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _mupdf.vectorq___delitem__(self, *args)

    def __getitem__(self, *args):
        return _mupdf.vectorq___getitem__(self, *args)

    def __setitem__(self, *args):
        return _mupdf.vectorq___setitem__(self, *args)

    def pop(self):
        return _mupdf.vectorq_pop(self)

    def append(self, x):
        return _mupdf.vectorq_append(self, x)

    def empty(self):
        return _mupdf.vectorq_empty(self)

    def size(self):
        return _mupdf.vectorq_size(self)

    def swap(self, v):
        return _mupdf.vectorq_swap(self, v)

    def begin(self):
        return _mupdf.vectorq_begin(self)

    def end(self):
        return _mupdf.vectorq_end(self)

    def rbegin(self):
        return _mupdf.vectorq_rbegin(self)

    def rend(self):
        return _mupdf.vectorq_rend(self)

    def clear(self):
        return _mupdf.vectorq_clear(self)

    def get_allocator(self):
        return _mupdf.vectorq_get_allocator(self)

    def pop_back(self):
        return _mupdf.vectorq_pop_back(self)

    def erase(self, *args):
        return _mupdf.vectorq_erase(self, *args)

    def __init__(self, *args):
        _mupdf.vectorq_swiginit(self, _mupdf.new_vectorq(*args))

    def push_back(self, x):
        return _mupdf.vectorq_push_back(self, x)

    def front(self):
        return _mupdf.vectorq_front(self)

    def back(self):
        return _mupdf.vectorq_back(self)

    def assign(self, n, x):
        return _mupdf.vectorq_assign(self, n, x)

    def resize(self, *args):
        return _mupdf.vectorq_resize(self, *args)

    def insert(self, *args):
        return _mupdf.vectorq_insert(self, *args)

    def reserve(self, n):
        return _mupdf.vectorq_reserve(self, n)

    def capacity(self):
        return _mupdf.vectorq_capacity(self)
    __swig_destroy__ = _mupdf.delete_vectorq