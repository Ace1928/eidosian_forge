import testtools
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import metadefs
def test_update_namespace_disallowed_fields(self):
    properties = {'display_name': 'My Updated Name'}
    self.controller.update(NAMESPACE1, **properties)
    actual = self.api.calls
    _disallowed_fields = ['self', 'schema', 'created_at', 'updated_at']
    for key in actual[1][3]:
        self.assertNotIn(key, _disallowed_fields)