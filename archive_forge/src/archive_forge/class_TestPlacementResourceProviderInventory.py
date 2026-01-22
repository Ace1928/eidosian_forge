from openstack.placement.v1 import _proxy
from openstack.placement.v1 import resource_class
from openstack.placement.v1 import resource_provider
from openstack.placement.v1 import resource_provider_inventory
from openstack.tests.unit import test_proxy_base as test_proxy_base
class TestPlacementResourceProviderInventory(TestPlacementProxy):

    def test_resource_provider_inventory_create(self):
        self.verify_create(self.proxy.create_resource_provider_inventory, resource_provider_inventory.ResourceProviderInventory, method_kwargs={'resource_provider': 'test_id', 'resource_class': 'CUSTOM_FOO', 'total': 20}, expected_kwargs={'resource_provider_id': 'test_id', 'resource_class': 'CUSTOM_FOO', 'total': 20})

    def test_resource_provider_inventory_delete(self):
        self.verify_delete(self.proxy.delete_resource_provider_inventory, resource_provider_inventory.ResourceProviderInventory, ignore_missing=False, method_kwargs={'resource_provider': 'test_id'}, expected_kwargs={'resource_provider_id': 'test_id'})

    def test_resource_provider_inventory_update(self):
        self.verify_update(self.proxy.update_resource_provider_inventory, resource_provider_inventory.ResourceProviderInventory, method_kwargs={'resource_provider': 'test_id', 'resource_provider_generation': 1}, expected_kwargs={'resource_provider_id': 'test_id', 'resource_provider_generation': 1})

    def test_resource_provider_inventory_get(self):
        self.verify_get(self.proxy.get_resource_provider_inventory, resource_provider_inventory.ResourceProviderInventory, method_kwargs={'resource_provider': 'test_id'}, expected_kwargs={'resource_provider_id': 'test_id'})

    def test_resource_provider_inventories(self):
        self.verify_list(self.proxy.resource_provider_inventories, resource_provider_inventory.ResourceProviderInventory, method_kwargs={'resource_provider': 'test_id'}, expected_kwargs={'resource_provider_id': 'test_id'})