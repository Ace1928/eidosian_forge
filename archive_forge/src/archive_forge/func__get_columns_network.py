from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib import utils
from osc_lib.utils import tags as _tag
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
def _get_columns_network(item):
    column_map = {'subnet_ids': 'subnets', 'is_admin_state_up': 'admin_state_up', 'is_router_external': 'router:external', 'is_port_security_enabled': 'port_security_enabled', 'provider_network_type': 'provider:network_type', 'provider_physical_network': 'provider:physical_network', 'provider_segmentation_id': 'provider:segmentation_id', 'is_shared': 'shared', 'ipv4_address_scope_id': 'ipv4_address_scope', 'ipv6_address_scope_id': 'ipv6_address_scope', 'tags': 'tags'}
    hidden_columns = ['location', 'tenant_id']
    hidden_columns = ['location']
    return utils.get_osc_show_columns_for_sdk_resource(item, column_map, hidden_columns)