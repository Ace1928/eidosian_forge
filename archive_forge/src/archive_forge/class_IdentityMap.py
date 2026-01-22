from . import errors
class IdentityMap:
    """An in memory map from object id to instance.

    An IdentityMap maps from keys to single instances of objects in memory.
    We have explicit calls on the map for the root of each inheritance tree
    that is store in the map. Look for find_CLASS and add_CLASS methods.
    """

    def add_weave(self, id, weave):
        """Add weave to the map with a given id."""
        if self._weave_key(id) in self._map:
            raise errors.BzrError('weave %s already in the identity map' % id)
        self._map[self._weave_key(id)] = weave
        self._reverse_map[weave] = self._weave_key(id)

    def find_weave(self, id):
        """Return the weave for 'id', or None if it is not present."""
        return self._map.get(self._weave_key(id), None)

    def __init__(self):
        super().__init__()
        self._map = {}
        self._reverse_map = {}

    def remove_object(self, an_object):
        """Remove object from map."""
        if isinstance(an_object, list):
            raise KeyError('%r not in identity map' % an_object)
        else:
            self._map.pop(self._reverse_map[an_object])
            self._reverse_map.pop(an_object)

    def _weave_key(self, id):
        """Return the key for a weaves id."""
        return 'weave-' + id