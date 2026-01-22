import logging
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
def _get_network_columns(item):
    column_map = {'is_admin_state_up': 'admin_state_up', 'is_alive': 'alive'}
    hidden_columns = ['location', 'name', 'tenant_id']
    return utils.get_osc_show_columns_for_sdk_resource(item, column_map, hidden_columns)