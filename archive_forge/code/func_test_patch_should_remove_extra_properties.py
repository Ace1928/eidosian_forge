import jsonpatch
import testtools
import warlock
from glanceclient.tests import utils
from glanceclient.v2 import schemas
def test_patch_should_remove_extra_properties(self):
    obj = {'name': 'fred', 'weight': '10'}
    original = self.model(obj)
    del original['weight']
    patch = original.patch
    expected = '[{"path": "/weight", "op": "remove"}]'
    self.assertTrue(compare_json_patches(patch, expected))