import testtools
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import metadefs
def test_update_object_disallowed_fields(self):
    properties = {'description': 'UPDATED_DESCRIPTION'}
    self.controller.update(NAMESPACE1, OBJECT1, **properties)
    actual = self.api.calls
    "('PUT', '/v2/metadefs/namespaces/Namespace1/objects/Object1', {},\n        [('description', 'UPDATED_DESCRIPTION'),\n        ('name', 'Object1'),\n        ('properties', ...),\n        ('required', [])])"
    _disallowed_fields = ['self', 'schema', 'created_at', 'updated_at']
    for key in actual[1][3]:
        self.assertNotIn(key, _disallowed_fields)