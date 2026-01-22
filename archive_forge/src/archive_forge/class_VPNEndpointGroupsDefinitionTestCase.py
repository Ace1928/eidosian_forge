from neutron_lib.api.definitions import vpn_endpoint_groups
from neutron_lib.tests.unit.api.definitions import base
class VPNEndpointGroupsDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = vpn_endpoint_groups
    extension_resources = ('endpoint_groups',)
    extension_attributes = ('type', 'endpoints')