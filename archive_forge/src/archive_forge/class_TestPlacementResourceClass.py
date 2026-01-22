from openstack.placement.v1 import _proxy
from openstack.placement.v1 import resource_class
from openstack.placement.v1 import resource_provider
from openstack.placement.v1 import resource_provider_inventory
from openstack.tests.unit import test_proxy_base as test_proxy_base
class TestPlacementResourceClass(TestPlacementProxy):

    def test_resource_class_create(self):
        self.verify_create(self.proxy.create_resource_class, resource_class.ResourceClass)

    def test_resource_class_delete(self):
        self.verify_delete(self.proxy.delete_resource_class, resource_class.ResourceClass, False)

    def test_resource_class_update(self):
        self.verify_update(self.proxy.update_resource_class, resource_class.ResourceClass, False)

    def test_resource_class_get(self):
        self.verify_get(self.proxy.get_resource_class, resource_class.ResourceClass)

    def test_resource_classes(self):
        self.verify_list(self.proxy.resource_classes, resource_class.ResourceClass)