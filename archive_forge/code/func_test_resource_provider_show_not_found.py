import operator
import uuid
from osc_placement.tests.functional import base
def test_resource_provider_show_not_found(self):
    rp_uuid = str(uuid.uuid4())
    msg = 'No resource provider with uuid ' + rp_uuid + ' found'
    exc = self.assertRaises(base.CommandException, self.resource_provider_show, rp_uuid)
    self.assertIn(msg, str(exc))