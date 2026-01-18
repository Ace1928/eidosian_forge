import uuid
from openstack.placement.v1 import resource_provider as _resource_provider
from openstack.tests.functional import base
def test_resource_provider_aggregates(self):
    aggregates = [uuid.uuid4().hex, uuid.uuid4().hex]
    resource_provider = self.operator_cloud.placement.set_resource_provider_aggregates(self.resource_provider, *aggregates)
    self.assertCountEqual(aggregates, resource_provider.aggregates)
    resource_provider = self.operator_cloud.placement.get_resource_provider_aggregates(self.resource_provider)
    self.assertCountEqual(aggregates, resource_provider.aggregates)