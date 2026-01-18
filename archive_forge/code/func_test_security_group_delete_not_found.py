from keystoneauth1 import session
from osc_lib import exceptions as osc_lib_exceptions
from requests_mock.contrib import fixture
from openstackclient.api import compute_v2 as compute
from openstackclient.tests.unit import utils
def test_security_group_delete_not_found(self):
    self.requests_mock.register_uri('GET', FAKE_URL + '/os-security-groups/sg3', status_code=404)
    self.requests_mock.register_uri('GET', FAKE_URL + '/os-security-groups', json={'security_groups': self.LIST_SECURITY_GROUP_RESP}, status_code=200)
    self.assertRaises(osc_lib_exceptions.NotFound, self.api.security_group_delete, 'sg3')