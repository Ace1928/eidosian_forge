from openstack.key_manager.v1 import _proxy
from openstack.key_manager.v1 import container
from openstack.key_manager.v1 import order
from openstack.key_manager.v1 import secret
from openstack.tests.unit import test_proxy_base
class TestKeyManagerProxy(test_proxy_base.TestProxyBase):

    def setUp(self):
        super(TestKeyManagerProxy, self).setUp()
        self.proxy = _proxy.Proxy(self.session)