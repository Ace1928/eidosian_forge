import uuid
from keystone.common import manager
from keystone.common import provider_api
from keystone.tests import unit
def test_provider_api_mixin(self):
    test_manager = self._create_manager_instance()

    class Testing(provider_api.ProviderAPIMixin, object):
        pass
    instance = Testing()
    self.assertIs(test_manager, getattr(instance, test_manager._provides_api))