import testtools
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import metadefs
def test_update_namespace_invalid_property(self):
    properties = {'protected': '123'}
    self.assertRaises(TypeError, self.controller.update, NAMESPACE1, **properties)