import sys
from . import errors as errors
from .identitymap import IdentityMap, NullIdentityMap
from .trace import mutter
class PassThroughTransaction(Transaction):
    """A pass through transaction

    - nothing is cached.
    - nothing ever gets into the identity map.
    """

    def finish(self):
        """Clean up this transaction."""
        for thing in self._dirty_objects:
            callback = getattr(thing, 'transaction_finished', None)
            if callback is not None:
                callback()

    def __init__(self):
        super().__init__()
        self.map = NullIdentityMap()
        self._dirty_objects = set()

    def register_clean(self, an_object, precious=False):
        """Register an_object as being clean.

        Note that precious is only a hint, and PassThroughTransaction
        ignores it.
        """

    def register_dirty(self, an_object):
        """Register an_object as being dirty.

        Dirty objects get informed
        when the transaction finishes.
        """
        self._dirty_objects.add(an_object)

    def set_cache_size(self, ignored):
        """Do nothing, we are passing through."""

    def writeable(self):
        """Pass through transactions allow writes."""
        return True