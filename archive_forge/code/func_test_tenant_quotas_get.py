from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_tenant_quotas_get(self):
    tenant_id = 'test'
    quota = cs.quotas.get(tenant_id)
    cs.assert_called('GET', '/os-quota-sets/%s?usage=False' % tenant_id)
    self._assert_request_id(quota)