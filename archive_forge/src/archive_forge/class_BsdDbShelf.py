from pickle import DEFAULT_PROTOCOL, Pickler, Unpickler
from io import BytesIO
import collections.abc
class BsdDbShelf(Shelf):
    """Shelf implementation using the "BSD" db interface.

    This adds methods first(), next(), previous(), last() and
    set_location() that have no counterpart in [g]dbm databases.

    The actual database must be opened using one of the "bsddb"
    modules "open" routines (i.e. bsddb.hashopen, bsddb.btopen or
    bsddb.rnopen) and passed to the constructor.

    See the module's __doc__ string for an overview of the interface.
    """

    def __init__(self, dict, protocol=None, writeback=False, keyencoding='utf-8'):
        Shelf.__init__(self, dict, protocol, writeback, keyencoding)

    def set_location(self, key):
        key, value = self.dict.set_location(key)
        f = BytesIO(value)
        return (key.decode(self.keyencoding), Unpickler(f).load())

    def next(self):
        key, value = next(self.dict)
        f = BytesIO(value)
        return (key.decode(self.keyencoding), Unpickler(f).load())

    def previous(self):
        key, value = self.dict.previous()
        f = BytesIO(value)
        return (key.decode(self.keyencoding), Unpickler(f).load())

    def first(self):
        key, value = self.dict.first()
        f = BytesIO(value)
        return (key.decode(self.keyencoding), Unpickler(f).load())

    def last(self):
        key, value = self.dict.last()
        f = BytesIO(value)
        return (key.decode(self.keyencoding), Unpickler(f).load())