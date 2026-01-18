from oslo_utils import encodeutils
from glance.common import exception
import glance.context
import glance.db
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_save_property(self):
    property = self.property_repo.get(NAMESPACE1, PROPERTY1)
    property.schema = '{"save": "schema"}'
    self.property_repo.save(property)
    property = self.property_repo.get(NAMESPACE1, PROPERTY1)
    self.assertEqual(PROPERTY1, property.name)
    self.assertEqual('{"save": "schema"}', property.schema)