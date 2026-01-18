from openstack.key_manager.v1 import _proxy
from openstack.key_manager.v1 import container
from openstack.key_manager.v1 import order
from openstack.key_manager.v1 import secret
from openstack.tests.unit import test_proxy_base
def test_server_create_attrs(self):
    self.verify_create(self.proxy.create_container, container.Container)