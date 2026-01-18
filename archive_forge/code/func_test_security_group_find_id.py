from keystoneauth1 import session
from osc_lib import exceptions as osc_lib_exceptions
from requests_mock.contrib import fixture
from openstackclient.api import compute_v2 as compute
from openstackclient.tests.unit import utils
def test_security_group_find_id(self):
    self.requests_mock.register_uri('GET', FAKE_URL + '/os-security-groups/1', json={'security_group': self.FAKE_SECURITY_GROUP_RESP}, status_code=200)
    ret = self.api.security_group_find('1')
    self.assertEqual(self.FAKE_SECURITY_GROUP_RESP, ret)