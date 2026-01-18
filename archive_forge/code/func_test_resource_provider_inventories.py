from openstack.placement.v1 import _proxy
from openstack.placement.v1 import resource_class
from openstack.placement.v1 import resource_provider
from openstack.placement.v1 import resource_provider_inventory
from openstack.tests.unit import test_proxy_base as test_proxy_base
def test_resource_provider_inventories(self):
    self.verify_list(self.proxy.resource_provider_inventories, resource_provider_inventory.ResourceProviderInventory, method_kwargs={'resource_provider': 'test_id'}, expected_kwargs={'resource_provider_id': 'test_id'})