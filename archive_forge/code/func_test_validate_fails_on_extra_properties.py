from glance.common import exception
import glance.schema
from glance.tests import utils as test_utils
def test_validate_fails_on_extra_properties(self):
    obj = {'ham': 'virginia', 'eggs': 'scrambled', 'bacon': 'crispy'}
    self.assertRaises(exception.InvalidObject, self.schema.validate, obj)