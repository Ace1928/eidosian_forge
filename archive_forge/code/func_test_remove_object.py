from .. import errors as errors
from .. import identitymap as identitymap
from . import TestCase
def test_remove_object(self):
    map = identitymap.IdentityMap()
    weave = 'foo'
    map.add_weave('id', weave)
    map.remove_object(weave)
    map.add_weave('id', weave)