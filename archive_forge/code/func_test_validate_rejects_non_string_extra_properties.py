from glance.common import exception
import glance.schema
from glance.tests import utils as test_utils
def test_validate_rejects_non_string_extra_properties(self):
    obj = {'ham': 'virginia', 'eggs': 'scrambled', 'grits': 1000}
    self.assertRaises(exception.InvalidObject, self.schema.validate, obj)