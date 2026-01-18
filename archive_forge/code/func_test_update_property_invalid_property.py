import testtools
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import metadefs
def test_update_property_invalid_property(self):
    properties = {'type': 'INVALID'}
    self.assertRaises(TypeError, self.controller.update, NAMESPACE1, PROPERTY1, **properties)