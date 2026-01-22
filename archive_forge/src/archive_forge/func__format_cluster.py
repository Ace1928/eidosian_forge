from cinderclient import api_versions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
def _format_cluster(cluster, detailed=False):
    columns = ('name', 'binary', 'state', 'status')
    column_headers = ('Name', 'Binary', 'State', 'Status')
    if detailed:
        columns += ('disabled_reason', 'num_hosts', 'num_down_hosts', 'last_heartbeat', 'created_at', 'updated_at', 'replication_status', 'frozen', 'active_backend_id')
        column_headers += ('Disabled Reason', 'Hosts', 'Down Hosts', 'Last Heartbeat', 'Created At', 'Updated At', 'Replication Status', 'Frozen', 'Active Backend ID')
    return (column_headers, utils.get_item_properties(cluster, columns))