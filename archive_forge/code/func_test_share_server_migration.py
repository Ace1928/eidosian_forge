import ast
import ddt
import testtools
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
@ddt.data('cancel', 'complete')
def test_share_server_migration(self, operation):
    share, share_network = self._create_share_and_share_network()
    share_server_id = share['share_server_id']
    src_host = share['host'].split('#')[0]
    pools = self.admin_client.pool_list(detail=True)
    host_list = list()
    for hosts in pools:
        host_name = hosts['Name'].split('#')[0]
        if ast.literal_eval(hosts['Capabilities']).get('driver_handles_share_servers') and host_name != src_host:
            host_list.append(host_name)
    host_list = list(set(host_list))
    if len(host_list) == 0:
        raise self.skipException('No hosts available for share server migration.')
    dest_backend = None
    for host in host_list:
        compatibility = self.admin_client.share_server_migration_check(server_id=share_server_id, dest_host=host, writable=False, nondisruptive=False, preserve_snapshots=False, new_share_network=None)
        if compatibility['compatible']:
            dest_host = host
    if dest_backend is not None:
        raise self.skipException('No hosts compatible to perform a share server migration.')
    self.admin_client.share_server_migration_start(share_server_id, dest_host)
    server = self.admin_client.get_share_server(share_server_id)
    share = self.admin_client.get_share(share['id'])
    self.assertEqual(constants.STATUS_SERVER_MIGRATING, share['status'])
    task_state = constants.TASK_STATE_MIGRATION_DRIVER_PHASE1_DONE
    server = self.admin_client.wait_for_server_migration_task_state(share_server_id, dest_host, task_state)
    migration_progress = self.admin_client.share_server_migration_get_progress(share_server_id)
    dest_share_server_id = migration_progress.get('destination_share_server_id')
    if operation == 'complete':
        task_state = constants.TASK_STATE_MIGRATION_SUCCESS
        self.admin_client.share_server_migration_complete(share_server_id)
        server = self.admin_client.wait_for_server_migration_task_state(dest_share_server_id, dest_host, task_state)
        self.admin_client.wait_for_share_server_deletion(share_server_id)
    else:
        self.admin_client.share_server_migration_cancel(server['id'])
        task_state = constants.TASK_STATE_MIGRATION_CANCELLED
        server = self.admin_client.wait_for_server_migration_task_state(server['id'], dest_host, task_state)
    share = self.admin_client.get_share(share['id'])
    self.assertEqual('available', share['status'])