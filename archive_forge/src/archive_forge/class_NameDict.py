import collections
import dns.name
from ._compat import xrange
class NameDict(collections.MutableMapping):
    """A dictionary whose keys are dns.name.Name objects.

    In addition to being like a regular Python dictionary, this
    dictionary can also get the deepest match for a given key.
    """
    __slots__ = ['max_depth', 'max_depth_items', '__store']

    def __init__(self, *args, **kwargs):
        super(NameDict, self).__init__()
        self.__store = dict()
        self.max_depth = 0
        self.max_depth_items = 0
        self.update(dict(*args, **kwargs))

    def __update_max_depth(self, key):
        if len(key) == self.max_depth:
            self.max_depth_items = self.max_depth_items + 1
        elif len(key) > self.max_depth:
            self.max_depth = len(key)
            self.max_depth_items = 1

    def __getitem__(self, key):
        return self.__store[key]

    def __setitem__(self, key, value):
        if not isinstance(key, dns.name.Name):
            raise ValueError('NameDict key must be a name')
        self.__store[key] = value
        self.__update_max_depth(key)

    def __delitem__(self, key):
        value = self.__store.pop(key)
        if len(value) == self.max_depth:
            self.max_depth_items = self.max_depth_items - 1
        if self.max_depth_items == 0:
            self.max_depth = 0
            for k in self.__store:
                self.__update_max_depth(k)

    def __iter__(self):
        return iter(self.__store)

    def __len__(self):
        return len(self.__store)

    def has_key(self, key):
        return key in self.__store

    def get_deepest_match(self, name):
        """Find the deepest match to *fname* in the dictionary.

        The deepest match is the longest name in the dictionary which is
        a superdomain of *name*.  Note that *superdomain* includes matching
        *name* itself.

        *name*, a ``dns.name.Name``, the name to find.

        Returns a ``(key, value)`` where *key* is the deepest
        ``dns.name.Name``, and *value* is the value associated with *key*.
        """
        depth = len(name)
        if depth > self.max_depth:
            depth = self.max_depth
        for i in xrange(-depth, 0):
            n = dns.name.Name(name[i:])
            if n in self:
                return (n, self[n])
        v = self[dns.name.empty]
        return (dns.name.empty, v)