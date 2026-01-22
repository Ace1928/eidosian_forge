import json
import re
from novaclient import exceptions as nova_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
def _get_hypervisor_columns(item, client):
    column_map = {'name': 'hypervisor_hostname'}
    hidden_columns = ['location', 'servers']
    if sdk_utils.supports_microversion(client, '2.88'):
        hidden_columns.extend(['current_workload', 'disk_available', 'local_disk_free', 'local_disk_size', 'local_disk_used', 'memory_free', 'memory_size', 'memory_used', 'running_vms', 'vcpus_used', 'vcpus'])
    else:
        column_map.update({'disk_available': 'disk_available_least', 'local_disk_free': 'free_disk_gb', 'local_disk_size': 'local_gb', 'local_disk_used': 'local_gb_used', 'memory_free': 'free_ram_mb', 'memory_used': 'memory_mb_used', 'memory_size': 'memory_mb'})
    return utils.get_osc_show_columns_for_sdk_resource(item, column_map, hidden_columns)