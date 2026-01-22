from sys import version_info as _swig_python_version_info
class LinOpVector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def iterator(self):
        return _cvxcore.LinOpVector_iterator(self)

    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _cvxcore.LinOpVector___nonzero__(self)

    def __bool__(self):
        return _cvxcore.LinOpVector___bool__(self)

    def __len__(self):
        return _cvxcore.LinOpVector___len__(self)

    def __getslice__(self, i, j):
        return _cvxcore.LinOpVector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _cvxcore.LinOpVector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _cvxcore.LinOpVector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _cvxcore.LinOpVector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _cvxcore.LinOpVector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _cvxcore.LinOpVector___setitem__(self, *args)

    def pop(self):
        return _cvxcore.LinOpVector_pop(self)

    def append(self, x):
        return _cvxcore.LinOpVector_append(self, x)

    def empty(self):
        return _cvxcore.LinOpVector_empty(self)

    def size(self):
        return _cvxcore.LinOpVector_size(self)

    def swap(self, v):
        return _cvxcore.LinOpVector_swap(self, v)

    def begin(self):
        return _cvxcore.LinOpVector_begin(self)

    def end(self):
        return _cvxcore.LinOpVector_end(self)

    def rbegin(self):
        return _cvxcore.LinOpVector_rbegin(self)

    def rend(self):
        return _cvxcore.LinOpVector_rend(self)

    def clear(self):
        return _cvxcore.LinOpVector_clear(self)

    def get_allocator(self):
        return _cvxcore.LinOpVector_get_allocator(self)

    def pop_back(self):
        return _cvxcore.LinOpVector_pop_back(self)

    def erase(self, *args):
        return _cvxcore.LinOpVector_erase(self, *args)

    def __init__(self, *args):
        _cvxcore.LinOpVector_swiginit(self, _cvxcore.new_LinOpVector(*args))

    def push_back(self, x):
        return _cvxcore.LinOpVector_push_back(self, x)

    def front(self):
        return _cvxcore.LinOpVector_front(self)

    def back(self):
        return _cvxcore.LinOpVector_back(self)

    def assign(self, n, x):
        return _cvxcore.LinOpVector_assign(self, n, x)

    def resize(self, *args):
        return _cvxcore.LinOpVector_resize(self, *args)

    def insert(self, *args):
        return _cvxcore.LinOpVector_insert(self, *args)

    def reserve(self, n):
        return _cvxcore.LinOpVector_reserve(self, n)

    def capacity(self):
        return _cvxcore.LinOpVector_capacity(self)
    __swig_destroy__ = _cvxcore.delete_LinOpVector