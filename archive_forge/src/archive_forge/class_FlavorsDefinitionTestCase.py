from neutron_lib.api.definitions import flavors
from neutron_lib.tests.unit.api.definitions import base
class FlavorsDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = flavors
    extension_resources = (flavors.FLAVORS, flavors.SERVICE_PROFILES)
    extension_subresources = (flavors.NEXT_PROVIDERS, flavors.SERVICE_PROFILES)
    extension_attributes = ('enabled', 'provider', 'metainfo', 'driver', flavors.SERVICE_PROFILES, 'service_type')