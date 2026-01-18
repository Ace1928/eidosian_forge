from keystoneauth1 import session
from osc_lib import exceptions as osc_lib_exceptions
from requests_mock.contrib import fixture
from openstackclient.api import compute_v2 as compute
from openstackclient.tests.unit import utils
def test_security_group_set_options_name(self):
    self.requests_mock.register_uri('GET', FAKE_URL + '/os-security-groups/sg2', status_code=404)
    self.requests_mock.register_uri('GET', FAKE_URL + '/os-security-groups', json={'security_groups': self.LIST_SECURITY_GROUP_RESP}, status_code=200)
    self.requests_mock.register_uri('PUT', FAKE_URL + '/os-security-groups/2', json={'security_group': self.FAKE_SECURITY_GROUP_RESP_2}, status_code=200)
    ret = self.api.security_group_set(security_group='sg2', description='desc2')
    self.assertEqual(self.FAKE_SECURITY_GROUP_RESP_2, ret)