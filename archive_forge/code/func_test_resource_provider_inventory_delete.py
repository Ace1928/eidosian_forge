from openstack.placement.v1 import _proxy
from openstack.placement.v1 import resource_class
from openstack.placement.v1 import resource_provider
from openstack.placement.v1 import resource_provider_inventory
from openstack.tests.unit import test_proxy_base as test_proxy_base
def test_resource_provider_inventory_delete(self):
    self.verify_delete(self.proxy.delete_resource_provider_inventory, resource_provider_inventory.ResourceProviderInventory, ignore_missing=False, method_kwargs={'resource_provider': 'test_id'}, expected_kwargs={'resource_provider_id': 'test_id'})