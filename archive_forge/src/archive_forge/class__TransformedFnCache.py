import inspect
import weakref
class _TransformedFnCache(object):
    """Generic hierarchical cache for transformed functions.

  The keys are soft references (i.e. they are discarded when the key is
  destroyed) created from the source function by `_get_key`. The subkeys are
  strong references and can be any value. Typically they identify different
  kinds of transformation.
  """
    __slots__ = ('_cache',)

    def __init__(self):
        self._cache = weakref.WeakKeyDictionary()

    def _get_key(self, entity):
        raise NotImplementedError('subclasses must override')

    def has(self, entity, subkey):
        key = self._get_key(entity)
        parent = self._cache.get(key, None)
        if parent is None:
            return False
        return subkey in parent

    def __getitem__(self, entity):
        key = self._get_key(entity)
        parent = self._cache.get(key, None)
        if parent is None:
            self._cache[key] = parent = {}
        return parent

    def __len__(self):
        return len(self._cache)