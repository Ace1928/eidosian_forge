import operator
import uuid
from osc_placement.tests.functional import base
def test_usage_empty(self):
    rp = self.resource_provider_create()
    self.assertEqual([], self.resource_provider_show_usage(rp['uuid']))