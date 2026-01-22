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
class vectors(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def iterator(self):
        return _mupdf.vectors_iterator(self)

    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _mupdf.vectors___nonzero__(self)

    def __bool__(self):
        return _mupdf.vectors___bool__(self)

    def __len__(self):
        return _mupdf.vectors___len__(self)

    def __getslice__(self, i, j):
        return _mupdf.vectors___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _mupdf.vectors___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _mupdf.vectors___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _mupdf.vectors___delitem__(self, *args)

    def __getitem__(self, *args):
        return _mupdf.vectors___getitem__(self, *args)

    def __setitem__(self, *args):
        return _mupdf.vectors___setitem__(self, *args)

    def pop(self):
        return _mupdf.vectors_pop(self)

    def append(self, x):
        return _mupdf.vectors_append(self, x)

    def empty(self):
        return _mupdf.vectors_empty(self)

    def size(self):
        return _mupdf.vectors_size(self)

    def swap(self, v):
        return _mupdf.vectors_swap(self, v)

    def begin(self):
        return _mupdf.vectors_begin(self)

    def end(self):
        return _mupdf.vectors_end(self)

    def rbegin(self):
        return _mupdf.vectors_rbegin(self)

    def rend(self):
        return _mupdf.vectors_rend(self)

    def clear(self):
        return _mupdf.vectors_clear(self)

    def get_allocator(self):
        return _mupdf.vectors_get_allocator(self)

    def pop_back(self):
        return _mupdf.vectors_pop_back(self)

    def erase(self, *args):
        return _mupdf.vectors_erase(self, *args)

    def __init__(self, *args):
        _mupdf.vectors_swiginit(self, _mupdf.new_vectors(*args))

    def push_back(self, x):
        return _mupdf.vectors_push_back(self, x)

    def front(self):
        return _mupdf.vectors_front(self)

    def back(self):
        return _mupdf.vectors_back(self)

    def assign(self, n, x):
        return _mupdf.vectors_assign(self, n, x)

    def resize(self, *args):
        return _mupdf.vectors_resize(self, *args)

    def insert(self, *args):
        return _mupdf.vectors_insert(self, *args)

    def reserve(self, n):
        return _mupdf.vectors_reserve(self, n)

    def capacity(self):
        return _mupdf.vectors_capacity(self)
    __swig_destroy__ = _mupdf.delete_vectors