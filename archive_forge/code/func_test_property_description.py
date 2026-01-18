import jsonpatch
import testtools
import warlock
from glanceclient.tests import utils
from glanceclient.v2 import schemas
def test_property_description(self):
    prop = schemas.SchemaProperty('size', description='some quantity')
    self.assertEqual('size', prop.name)
    self.assertEqual('some quantity', prop.description)