from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.network.v2 import port
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
@staticmethod
def _get_common_cols_data(fake_port):
    columns = ('admin_state_up', 'allowed_address_pairs', 'binding_host_id', 'binding_profile', 'binding_vif_details', 'binding_vif_type', 'binding_vnic_type', 'created_at', 'data_plane_status', 'description', 'device_id', 'device_owner', 'device_profile', 'dns_assignment', 'dns_domain', 'dns_name', 'extra_dhcp_opts', 'fixed_ips', 'hardware_offload_type', 'hints', 'id', 'ip_allocation', 'mac_address', 'name', 'network_id', 'numa_affinity_policy', 'port_security_enabled', 'project_id', 'propagate_uplink_status', 'resource_request', 'revision_number', 'qos_network_policy_id', 'qos_policy_id', 'security_group_ids', 'status', 'tags', 'trunk_details', 'updated_at')
    data = (port.AdminStateColumn(fake_port.is_admin_state_up), format_columns.ListDictColumn(fake_port.allowed_address_pairs), fake_port.binding_host_id, format_columns.DictColumn(fake_port.binding_profile), format_columns.DictColumn(fake_port.binding_vif_details), fake_port.binding_vif_type, fake_port.binding_vnic_type, fake_port.created_at, fake_port.data_plane_status, fake_port.description, fake_port.device_id, fake_port.device_owner, fake_port.device_profile, format_columns.ListDictColumn(fake_port.dns_assignment), fake_port.dns_domain, fake_port.dns_name, format_columns.ListDictColumn(fake_port.extra_dhcp_opts), format_columns.ListDictColumn(fake_port.fixed_ips), fake_port.hardware_offload_type, fake_port.hints, fake_port.id, fake_port.ip_allocation, fake_port.mac_address, fake_port.name, fake_port.network_id, fake_port.numa_affinity_policy, fake_port.is_port_security_enabled, fake_port.project_id, fake_port.propagate_uplink_status, fake_port.resource_request, fake_port.revision_number, fake_port.qos_network_policy_id, fake_port.qos_policy_id, format_columns.ListColumn(fake_port.security_group_ids), fake_port.status, format_columns.ListColumn(fake_port.tags), fake_port.trunk_details, fake_port.updated_at)
    return (columns, data)