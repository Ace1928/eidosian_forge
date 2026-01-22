from pickle import DEFAULT_PROTOCOL, Pickler, Unpickler
from io import BytesIO
import collections.abc
class DbfilenameShelf(Shelf):
    """Shelf implementation using the "dbm" generic dbm interface.

    This is initialized with the filename for the dbm database.
    See the module's __doc__ string for an overview of the interface.
    """

    def __init__(self, filename, flag='c', protocol=None, writeback=False):
        import dbm
        Shelf.__init__(self, dbm.open(filename, flag), protocol, writeback)