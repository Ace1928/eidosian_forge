import operator
import uuid
from osc_placement.tests.functional import base
def test_resource_provider_set_not_found(self):
    rp_uuid = str(uuid.uuid4())
    msg = 'No resource provider with uuid ' + rp_uuid + ' found'
    exc = self.assertRaises(base.CommandException, self.resource_provider_set, rp_uuid, 'test')
    self.assertIn(msg, str(exc))