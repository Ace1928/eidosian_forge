import jsonpatch
import testtools
import warlock
from glanceclient.tests import utils
from glanceclient.v2 import schemas
def test_property_is_base(self):
    raw_schema = {'name': 'Country', 'properties': {'size': {}, 'population': {'is_base': False}}}
    schema = schemas.Schema(raw_schema)
    self.assertTrue(schema.is_base_property('size'))
    self.assertFalse(schema.is_base_property('population'))
    self.assertFalse(schema.is_base_property('foo'))