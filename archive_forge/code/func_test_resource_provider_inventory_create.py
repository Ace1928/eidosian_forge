from openstack.placement.v1 import _proxy
from openstack.placement.v1 import resource_class
from openstack.placement.v1 import resource_provider
from openstack.placement.v1 import resource_provider_inventory
from openstack.tests.unit import test_proxy_base as test_proxy_base
def test_resource_provider_inventory_create(self):
    self.verify_create(self.proxy.create_resource_provider_inventory, resource_provider_inventory.ResourceProviderInventory, method_kwargs={'resource_provider': 'test_id', 'resource_class': 'CUSTOM_FOO', 'total': 20}, expected_kwargs={'resource_provider_id': 'test_id', 'resource_class': 'CUSTOM_FOO', 'total': 20})