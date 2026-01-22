from neutron_lib.api.definitions import network_ip_availability
from neutron_lib.tests.unit.api.definitions import base
class NetworkIPAvailabilityDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = network_ip_availability
    extension_resources = (network_ip_availability.RESOURCE_PLURAL,)
    extension_attributes = ('total_ips', 'used_ips', 'subnet_ip_availability', 'network_name')