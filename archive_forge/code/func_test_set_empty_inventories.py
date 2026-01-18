import collections
import copy
import uuid
from osc_placement.tests.functional import base
def test_set_empty_inventories(self):
    rp = self.resource_provider_create()
    self.assertEqual([], self.resource_inventory_set(rp['uuid']))