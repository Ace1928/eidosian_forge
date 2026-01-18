import testtools
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import metadefs
def test_update_tag(self):
    properties = {'name': TAG2}
    tag = self.controller.update(NAMESPACE1, TAG1, **properties)
    self.assertEqual(TAG2, tag.name)