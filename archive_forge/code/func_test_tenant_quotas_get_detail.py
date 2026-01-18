from novaclient import api_versions
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit.fixture_data import quotas as data
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
def test_tenant_quotas_get_detail(self):
    tenant_id = 'test'
    q = self.cs.quotas.get(tenant_id, detail=True)
    self.assert_request_id(q, fakes.FAKE_REQUEST_ID_LIST)
    self.assert_called('GET', '/os-quota-sets/%s/detail' % tenant_id)