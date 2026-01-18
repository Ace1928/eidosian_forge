import testtools
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import metadefs
def test_update_property_disallowed_fields(self):
    properties = {'description': 'UPDATED_DESCRIPTION'}
    self.controller.update(NAMESPACE1, PROPERTY1, **properties)
    actual = self.api.calls
    _disallowed_fields = ['created_at', 'updated_at']
    for key in actual[1][3]:
        self.assertNotIn(key, _disallowed_fields)