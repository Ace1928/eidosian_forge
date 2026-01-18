from .. import errors as errors
from .. import identitymap as identitymap
from . import TestCase
def test_null_identity_map_has_no_remove(self):
    map = identitymap.NullIdentityMap()
    self.assertEqual(None, getattr(map, 'remove_object', None))