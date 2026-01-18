import jsonpatch
import testtools
import warlock
from glanceclient.tests import utils
from glanceclient.v2 import schemas
def test_patch_should_remove_core_properties(self):
    obj = {'name': 'fred', 'color': 'red'}
    original = self.model(obj)
    del original['color']
    patch = original.patch
    expected = '[{"path": "/color", "op": "remove"}]'
    self.assertTrue(compare_json_patches(patch, expected))