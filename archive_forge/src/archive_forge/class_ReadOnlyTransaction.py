import sys
from . import errors as errors
from .identitymap import IdentityMap, NullIdentityMap
from .trace import mutter
class ReadOnlyTransaction(Transaction):
    """A read only unit of work for data objects."""

    def finish(self):
        """Clean up this transaction."""

    def __init__(self):
        super().__init__()
        self.map = IdentityMap()
        self._clean_objects = set()
        self._clean_queue = []
        self._limit = -1
        self._precious_objects = set()

    def is_clean(self, an_object):
        """Return True if an_object is clean."""
        return an_object in self._clean_objects

    def register_clean(self, an_object, precious=False):
        """Register an_object as being clean.

        If the precious hint is True, the object will not
        be ejected from the object identity map ever.
        """
        self._clean_objects.add(an_object)
        self._clean_queue.append(an_object)
        if precious:
            self._precious_objects.add(an_object)
        self._trim()

    def register_dirty(self, an_object):
        """Register an_object as being dirty."""
        raise errors.ReadOnlyObjectDirtiedError(an_object)

    def set_cache_size(self, size):
        """Set a new cache size."""
        if size < -1:
            raise ValueError(size)
        self._limit = size
        self._trim()

    def _trim(self):
        """Trim the cache back if needed."""
        if self._limit < 0 or self._limit - len(self._clean_objects) > 0:
            return
        needed = len(self._clean_objects) - self._limit
        offset = 0
        while needed and offset < len(self._clean_objects):
            if sys.version_info >= (3, 11):
                ref_threshold = 6
            else:
                ref_threshold = 7
            if sys.getrefcount(self._clean_queue[offset]) <= ref_threshold and (not self._clean_queue[offset] in self._precious_objects):
                removed = self._clean_queue[offset]
                self._clean_objects.remove(removed)
                del self._clean_queue[offset]
                self.map.remove_object(removed)
                mutter('removed object %r', removed)
                needed -= 1
            else:
                offset += 1

    def writeable(self):
        """Read only transactions do not allow writes."""
        return False