import operator
import uuid
from osc_placement.tests.functional import base
def test_resource_provider_show_allocations_empty(self):
    created = self.resource_provider_create()
    expected = dict(created, allocations={})
    retrieved = self.resource_provider_show(created['uuid'], allocations=True)
    self.assertEqual(expected, retrieved)