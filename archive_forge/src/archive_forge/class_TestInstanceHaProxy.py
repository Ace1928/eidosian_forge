from openstack.instance_ha.v1 import _proxy
from openstack.instance_ha.v1 import host
from openstack.instance_ha.v1 import notification
from openstack.instance_ha.v1 import segment
from openstack.instance_ha.v1 import vmove
from openstack.tests.unit import test_proxy_base
class TestInstanceHaProxy(test_proxy_base.TestProxyBase):

    def setUp(self):
        super(TestInstanceHaProxy, self).setUp()
        self.proxy = _proxy.Proxy(self.session)