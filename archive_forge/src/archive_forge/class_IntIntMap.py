from sys import version_info as _swig_python_version_info
class IntIntMap(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def iterator(self):
        return _cvxcore.IntIntMap_iterator(self)

    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _cvxcore.IntIntMap___nonzero__(self)

    def __bool__(self):
        return _cvxcore.IntIntMap___bool__(self)

    def __len__(self):
        return _cvxcore.IntIntMap___len__(self)

    def __iter__(self):
        return self.key_iterator()

    def iterkeys(self):
        return self.key_iterator()

    def itervalues(self):
        return self.value_iterator()

    def iteritems(self):
        return self.iterator()

    def __getitem__(self, key):
        return _cvxcore.IntIntMap___getitem__(self, key)

    def __delitem__(self, key):
        return _cvxcore.IntIntMap___delitem__(self, key)

    def has_key(self, key):
        return _cvxcore.IntIntMap_has_key(self, key)

    def keys(self):
        return _cvxcore.IntIntMap_keys(self)

    def values(self):
        return _cvxcore.IntIntMap_values(self)

    def items(self):
        return _cvxcore.IntIntMap_items(self)

    def __contains__(self, key):
        return _cvxcore.IntIntMap___contains__(self, key)

    def key_iterator(self):
        return _cvxcore.IntIntMap_key_iterator(self)

    def value_iterator(self):
        return _cvxcore.IntIntMap_value_iterator(self)

    def __setitem__(self, *args):
        return _cvxcore.IntIntMap___setitem__(self, *args)

    def asdict(self):
        return _cvxcore.IntIntMap_asdict(self)

    def __init__(self, *args):
        _cvxcore.IntIntMap_swiginit(self, _cvxcore.new_IntIntMap(*args))

    def empty(self):
        return _cvxcore.IntIntMap_empty(self)

    def size(self):
        return _cvxcore.IntIntMap_size(self)

    def swap(self, v):
        return _cvxcore.IntIntMap_swap(self, v)

    def begin(self):
        return _cvxcore.IntIntMap_begin(self)

    def end(self):
        return _cvxcore.IntIntMap_end(self)

    def rbegin(self):
        return _cvxcore.IntIntMap_rbegin(self)

    def rend(self):
        return _cvxcore.IntIntMap_rend(self)

    def clear(self):
        return _cvxcore.IntIntMap_clear(self)

    def get_allocator(self):
        return _cvxcore.IntIntMap_get_allocator(self)

    def count(self, x):
        return _cvxcore.IntIntMap_count(self, x)

    def erase(self, *args):
        return _cvxcore.IntIntMap_erase(self, *args)

    def find(self, x):
        return _cvxcore.IntIntMap_find(self, x)

    def lower_bound(self, x):
        return _cvxcore.IntIntMap_lower_bound(self, x)

    def upper_bound(self, x):
        return _cvxcore.IntIntMap_upper_bound(self, x)
    __swig_destroy__ = _cvxcore.delete_IntIntMap