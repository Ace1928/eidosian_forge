from glance.common import exception
import glance.schema
from glance.tests import utils as test_utils
class TestPermissiveSchema(test_utils.BaseTestCase):

    def setUp(self):
        super(TestPermissiveSchema, self).setUp()
        properties = {'ham': {'type': 'string'}, 'eggs': {'type': 'string'}}
        self.schema = glance.schema.PermissiveSchema('permissive', properties)

    def test_validate_with_additional_properties_allowed(self):
        obj = {'ham': 'virginia', 'eggs': 'scrambled', 'bacon': 'crispy'}
        self.schema.validate(obj)

    def test_validate_rejects_non_string_extra_properties(self):
        obj = {'ham': 'virginia', 'eggs': 'scrambled', 'grits': 1000}
        self.assertRaises(exception.InvalidObject, self.schema.validate, obj)

    def test_filter_passes_extra_properties(self):
        obj = {'ham': 'virginia', 'eggs': 'scrambled', 'bacon': 'crispy'}
        filtered = self.schema.filter(obj)
        self.assertEqual(obj, filtered)

    def test_raw_json_schema(self):
        expected = {'name': 'permissive', 'properties': {'ham': {'type': 'string'}, 'eggs': {'type': 'string'}}, 'additionalProperties': {'type': 'string'}}
        self.assertEqual(expected, self.schema.raw())