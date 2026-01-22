from openstack.placement.v1 import _proxy
from openstack.placement.v1 import resource_class
from openstack.placement.v1 import resource_provider
from openstack.placement.v1 import resource_provider_inventory
from openstack.tests.unit import test_proxy_base as test_proxy_base
class TestPlacementResourceProvider(TestPlacementProxy):

    def test_resource_provider_create(self):
        self.verify_create(self.proxy.create_resource_provider, resource_provider.ResourceProvider)

    def test_resource_provider_delete(self):
        self.verify_delete(self.proxy.delete_resource_provider, resource_provider.ResourceProvider, False)

    def test_resource_provider_update(self):
        self.verify_update(self.proxy.update_resource_provider, resource_provider.ResourceProvider, False)

    def test_resource_provider_get(self):
        self.verify_get(self.proxy.get_resource_provider, resource_provider.ResourceProvider)

    def test_resource_providers(self):
        self.verify_list(self.proxy.resource_providers, resource_provider.ResourceProvider)

    def test_resource_provider_set_aggregates(self):
        self._verify('openstack.placement.v1.resource_provider.ResourceProvider.set_aggregates', self.proxy.set_resource_provider_aggregates, method_args=['value', 'a', 'b'], expected_args=[self.proxy], expected_kwargs={'aggregates': ('a', 'b')})

    def test_resource_provider_get_aggregates(self):
        self._verify('openstack.placement.v1.resource_provider.ResourceProvider.fetch_aggregates', self.proxy.get_resource_provider_aggregates, method_args=['value'], expected_args=[self.proxy])