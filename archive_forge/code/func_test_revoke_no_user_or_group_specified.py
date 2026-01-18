import testtools
from testtools import matchers
from openstack import exceptions
from openstack.tests.unit import base
def test_revoke_no_user_or_group_specified(self):
    uris = self.__get('role', self.role_data, 'role_name', [], use_name=True)
    self.register_uris(uris)
    with testtools.ExpectedException(exceptions.SDKException, 'Must specify either a user or a group'):
        self.cloud.revoke_role(self.role_data.role_name)
    self.assert_calls()