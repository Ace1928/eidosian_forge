from unittest import mock
from keystoneauth1 import adapter
from openstack.compute.v2 import flavor
from openstack.tests.unit import base
def test_remove_tenant_access(self):
    sot = flavor.Flavor(**BASIC_EXAMPLE)
    resp = mock.Mock()
    resp.body = None
    resp.json = mock.Mock(return_value=resp.body)
    resp.status_code = 200
    self.sess.post = mock.Mock(return_value=resp)
    sot.remove_tenant_access(self.sess, 'fake_tenant')
    self.sess.post.assert_called_with('flavors/IDENTIFIER/action', json={'removeTenantAccess': {'tenant': 'fake_tenant'}}, headers={'Accept': ''})