from sys import version_info as _swig_python_version_info
class DoubleVector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def iterator(self):
        return _cvxcore.DoubleVector_iterator(self)

    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _cvxcore.DoubleVector___nonzero__(self)

    def __bool__(self):
        return _cvxcore.DoubleVector___bool__(self)

    def __len__(self):
        return _cvxcore.DoubleVector___len__(self)

    def __getslice__(self, i, j):
        return _cvxcore.DoubleVector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _cvxcore.DoubleVector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _cvxcore.DoubleVector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _cvxcore.DoubleVector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _cvxcore.DoubleVector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _cvxcore.DoubleVector___setitem__(self, *args)

    def pop(self):
        return _cvxcore.DoubleVector_pop(self)

    def append(self, x):
        return _cvxcore.DoubleVector_append(self, x)

    def empty(self):
        return _cvxcore.DoubleVector_empty(self)

    def size(self):
        return _cvxcore.DoubleVector_size(self)

    def swap(self, v):
        return _cvxcore.DoubleVector_swap(self, v)

    def begin(self):
        return _cvxcore.DoubleVector_begin(self)

    def end(self):
        return _cvxcore.DoubleVector_end(self)

    def rbegin(self):
        return _cvxcore.DoubleVector_rbegin(self)

    def rend(self):
        return _cvxcore.DoubleVector_rend(self)

    def clear(self):
        return _cvxcore.DoubleVector_clear(self)

    def get_allocator(self):
        return _cvxcore.DoubleVector_get_allocator(self)

    def pop_back(self):
        return _cvxcore.DoubleVector_pop_back(self)

    def erase(self, *args):
        return _cvxcore.DoubleVector_erase(self, *args)

    def __init__(self, *args):
        _cvxcore.DoubleVector_swiginit(self, _cvxcore.new_DoubleVector(*args))

    def push_back(self, x):
        return _cvxcore.DoubleVector_push_back(self, x)

    def front(self):
        return _cvxcore.DoubleVector_front(self)

    def back(self):
        return _cvxcore.DoubleVector_back(self)

    def assign(self, n, x):
        return _cvxcore.DoubleVector_assign(self, n, x)

    def resize(self, *args):
        return _cvxcore.DoubleVector_resize(self, *args)

    def insert(self, *args):
        return _cvxcore.DoubleVector_insert(self, *args)

    def reserve(self, n):
        return _cvxcore.DoubleVector_reserve(self, n)

    def capacity(self):
        return _cvxcore.DoubleVector_capacity(self)
    __swig_destroy__ = _cvxcore.delete_DoubleVector