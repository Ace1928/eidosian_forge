from osc_lib.command import command
from osc_lib import utils
from openstackclient.i18n import _
def _get_hypervisor_stat_columns(item):
    column_map = {'disk_available': 'disk_available_least', 'local_disk_free': 'free_disk_gb', 'local_disk_size': 'local_gb', 'local_disk_used': 'local_gb_used', 'memory_free': 'free_ram_mb', 'memory_size': 'memory_mb', 'memory_used': 'memory_mb_used'}
    hidden_columns = ['id', 'links', 'location', 'name']
    return utils.get_osc_show_columns_for_sdk_resource(item, column_map, hidden_columns)