import jsonpatch
import testtools
import warlock
from glanceclient.tests import utils
from glanceclient.v2 import schemas
def test_patch_should_replace_tags(self):
    obj = {'name': 'fred'}
    original = self.model(obj)
    original['tags'] = ['tag1', 'tag2']
    patch = original.patch
    expected = '[{"path": "/tags", "value": ["tag1", "tag2"], "op": "replace"}]'
    self.assertTrue(compare_json_patches(patch, expected))