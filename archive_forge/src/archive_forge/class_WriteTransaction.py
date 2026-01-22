import sys
from . import errors as errors
from .identitymap import IdentityMap, NullIdentityMap
from .trace import mutter
class WriteTransaction(ReadOnlyTransaction):
    """A write transaction

    - caches domain objects
    - clean objects can be removed from the cache
    - dirty objects are retained.
    """

    def finish(self):
        """Clean up this transaction."""
        for thing in self._dirty_objects:
            callback = getattr(thing, 'transaction_finished', None)
            if callback is not None:
                callback()

    def __init__(self):
        super().__init__()
        self._dirty_objects = set()

    def is_dirty(self, an_object):
        """Return True if an_object is dirty."""
        return an_object in self._dirty_objects

    def register_dirty(self, an_object):
        """Register an_object as being dirty.

        Dirty objects are not ejected from the identity map
        until the transaction finishes and get informed
        when the transaction finishes.
        """
        self._dirty_objects.add(an_object)
        if self.is_clean(an_object):
            self._clean_objects.remove(an_object)
            del self._clean_queue[self._clean_queue.index(an_object)]
        self._trim()

    def writeable(self):
        """Write transactions allow writes."""
        return True