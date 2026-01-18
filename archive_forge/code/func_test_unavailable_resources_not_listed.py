from heatclient import exc
import keystoneclient
from heat_integrationtests.functional import functional_base
def test_unavailable_resources_not_listed(self):
    resources = self.client.resource_types.list()
    self.assertFalse(any((self.unavailable_service in r.resource_type for r in resources)))