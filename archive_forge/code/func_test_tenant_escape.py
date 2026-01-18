from heat.common import identifier
from heat.tests import common
def test_tenant_escape(self):
    hi = identifier.HeatIdentifier(':/', 's', 'i')
    self.assertEqual(':/', hi.tenant)
    self.assertEqual('%3A%2F/stacks/s/i', hi.url_path())
    self.assertEqual('arn:openstack:heat::%3A%2F:stacks/s/i', hi.arn())