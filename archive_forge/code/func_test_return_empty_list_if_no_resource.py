import operator
import uuid
from osc_placement.tests.functional import base
def test_return_empty_list_if_no_resource(self):
    rp = self.resource_provider_create()
    self.assertEqual([], self.resource_provider_list(resources=['MEMORY_MB=256'], uuid=rp['uuid']))