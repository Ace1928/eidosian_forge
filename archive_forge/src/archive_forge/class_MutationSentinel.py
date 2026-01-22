import collections
import weakref
from tensorflow.python.util import object_identity
class MutationSentinel(object):
    """Container for tracking whether a property is in a cached state."""
    _in_cached_state = False

    def mark_as(self, value):
        may_affect_upstream = value != self._in_cached_state
        self._in_cached_state = value
        return may_affect_upstream

    @property
    def in_cached_state(self):
        return self._in_cached_state