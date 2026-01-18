import jsonpatch
import testtools
import warlock
from glanceclient.tests import utils
from glanceclient.v2 import schemas
def test_patch_should_replace_custom_properties(self):
    obj = {'name': 'fred', 'shape': 'circle'}
    original = self.model(obj)
    original['shape'] = 'square'
    patch = original.patch
    expected = '[{"path": "/shape", "value": "square", "op": "replace"}]'
    self.assertTrue(compare_json_patches(patch, expected))