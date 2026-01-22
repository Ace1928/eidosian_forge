from sys import version_info as _swig_python_version_info
import weakref
class SimpleBoundCosts(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self, num_bounds, default_bound_cost):
        _pywrapcp.SimpleBoundCosts_swiginit(self, _pywrapcp.new_SimpleBoundCosts(num_bounds, default_bound_cost))

    def bound_cost(self, element):
        return _pywrapcp.SimpleBoundCosts_bound_cost(self, element)

    def size(self):
        return _pywrapcp.SimpleBoundCosts_size(self)
    __swig_destroy__ = _pywrapcp.delete_SimpleBoundCosts