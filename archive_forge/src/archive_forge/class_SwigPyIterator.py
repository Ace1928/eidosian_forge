from sys import version_info as _swig_python_version_info
class SwigPyIterator(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')

    def __init__(self, *args, **kwargs):
        raise AttributeError('No constructor defined - class is abstract')
    __repr__ = _swig_repr
    __swig_destroy__ = _cvxcore.delete_SwigPyIterator

    def value(self):
        return _cvxcore.SwigPyIterator_value(self)

    def incr(self, n=1):
        return _cvxcore.SwigPyIterator_incr(self, n)

    def decr(self, n=1):
        return _cvxcore.SwigPyIterator_decr(self, n)

    def distance(self, x):
        return _cvxcore.SwigPyIterator_distance(self, x)

    def equal(self, x):
        return _cvxcore.SwigPyIterator_equal(self, x)

    def copy(self):
        return _cvxcore.SwigPyIterator_copy(self)

    def next(self):
        return _cvxcore.SwigPyIterator_next(self)

    def __next__(self):
        return _cvxcore.SwigPyIterator___next__(self)

    def previous(self):
        return _cvxcore.SwigPyIterator_previous(self)

    def advance(self, n):
        return _cvxcore.SwigPyIterator_advance(self, n)

    def __eq__(self, x):
        return _cvxcore.SwigPyIterator___eq__(self, x)

    def __ne__(self, x):
        return _cvxcore.SwigPyIterator___ne__(self, x)

    def __iadd__(self, n):
        return _cvxcore.SwigPyIterator___iadd__(self, n)

    def __isub__(self, n):
        return _cvxcore.SwigPyIterator___isub__(self, n)

    def __add__(self, n):
        return _cvxcore.SwigPyIterator___add__(self, n)

    def __sub__(self, *args):
        return _cvxcore.SwigPyIterator___sub__(self, *args)

    def __iter__(self):
        return self