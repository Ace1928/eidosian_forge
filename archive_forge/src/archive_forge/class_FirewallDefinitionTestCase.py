from neutron_lib.api.definitions import firewall_v2
from neutron_lib import constants
from neutron_lib.tests.unit.api.definitions import base
class FirewallDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = firewall_v2
    extension_resources = ('firewall_groups', 'firewall_policies', 'firewall_rules')
    extension_attributes = ('action', 'admin_state_up', 'audited', 'destination_ip_address', 'destination_port', 'egress_firewall_policy_id', 'enabled', 'firewall_policy_id', 'firewall_rules', 'ingress_firewall_policy_id', 'ip_version', 'ports', 'position', 'protocol', constants.SHARED, 'source_ip_address', 'source_port', 'source_firewall_group_id', 'destination_firewall_group_id')