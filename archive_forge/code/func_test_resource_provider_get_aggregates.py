from openstack.placement.v1 import _proxy
from openstack.placement.v1 import resource_class
from openstack.placement.v1 import resource_provider
from openstack.placement.v1 import resource_provider_inventory
from openstack.tests.unit import test_proxy_base as test_proxy_base
def test_resource_provider_get_aggregates(self):
    self._verify('openstack.placement.v1.resource_provider.ResourceProvider.fetch_aggregates', self.proxy.get_resource_provider_aggregates, method_args=['value'], expected_args=[self.proxy])