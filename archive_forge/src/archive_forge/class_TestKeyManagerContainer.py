from openstack.key_manager.v1 import _proxy
from openstack.key_manager.v1 import container
from openstack.key_manager.v1 import order
from openstack.key_manager.v1 import secret
from openstack.tests.unit import test_proxy_base
class TestKeyManagerContainer(TestKeyManagerProxy):

    def test_server_create_attrs(self):
        self.verify_create(self.proxy.create_container, container.Container)

    def test_container_delete(self):
        self.verify_delete(self.proxy.delete_container, container.Container, False)

    def test_container_delete_ignore(self):
        self.verify_delete(self.proxy.delete_container, container.Container, True)

    def test_container_find(self):
        self.verify_find(self.proxy.find_container, container.Container)

    def test_container_get(self):
        self.verify_get(self.proxy.get_container, container.Container)

    def test_containers(self):
        self.verify_list(self.proxy.containers, container.Container)

    def test_container_update(self):
        self.verify_update(self.proxy.update_container, container.Container)