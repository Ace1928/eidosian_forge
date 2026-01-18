from glance.common import exception
import glance.schema
from glance.tests import utils as test_utils
def test_validate_fails_on_bad_type(self):
    obj = {'eggs': 2}
    self.assertRaises(exception.InvalidObject, self.schema.validate, obj)