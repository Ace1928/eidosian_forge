from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_tenant_quotas_defaults(self):
    tenant_id = 'test'
    quota = cs.quotas.defaults(tenant_id)
    cs.assert_called('GET', '/os-quota-sets/%s/defaults' % tenant_id)
    self._assert_request_id(quota)