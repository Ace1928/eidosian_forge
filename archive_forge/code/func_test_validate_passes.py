from glance.common import exception
import glance.schema
from glance.tests import utils as test_utils
def test_validate_passes(self):
    obj = {'ham': 'no', 'eggs': 'scrambled'}
    self.schema.validate(obj)