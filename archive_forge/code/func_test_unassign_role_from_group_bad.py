from unittest import mock
from keystoneauth1 import adapter
from openstack.identity.v3 import domain
from openstack.identity.v3 import group
from openstack.identity.v3 import role
from openstack.identity.v3 import user
from openstack.tests.unit import base
def test_unassign_role_from_group_bad(self):
    sot = domain.Domain(**EXAMPLE)
    resp = self.bad_resp
    self.sess.delete = mock.Mock(return_value=resp)
    self.assertFalse(sot.unassign_role_from_group(self.sess, group.Group(id='1'), role.Role(id='2')))