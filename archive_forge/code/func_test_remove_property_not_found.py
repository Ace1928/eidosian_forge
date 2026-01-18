from oslo_utils import encodeutils
from glance.common import exception
import glance.context
import glance.db
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_remove_property_not_found(self):
    fake_name = 'fake_name'
    property = self.property_repo.get(NAMESPACE1, PROPERTY1)
    property.name = fake_name
    self.assertRaises(exception.NotFound, self.property_repo.remove, property)