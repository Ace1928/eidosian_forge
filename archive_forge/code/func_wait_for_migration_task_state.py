import ast
import re
import time
from oslo_utils import strutils
from tempest.lib.cli import base
from tempest.lib.cli import output_parser
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions as tempest_lib_exc
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import exceptions
from manilaclient.tests.functional import utils
def wait_for_migration_task_state(self, share_id, dest_host, task_state_to_wait, microversion=None):
    """Waits for a share to migrate to a certain host."""
    statuses = (task_state_to_wait,) if not isinstance(task_state_to_wait, (tuple, list, set)) else task_state_to_wait
    share = self.get_share(share_id, microversion=microversion)
    start = int(time.time())
    while share['task_state'] not in statuses:
        time.sleep(self.build_interval)
        share = self.get_share(share_id, microversion=microversion)
        if share['task_state'] in statuses:
            break
        elif share['task_state'] == constants.TASK_STATE_MIGRATION_ERROR:
            raise exceptions.ShareMigrationException(share_id=share['id'], src=share['host'], dest=dest_host)
        elif int(time.time()) - start >= self.build_timeout:
            message = 'Share %(share_id)s failed to reach a status in%(status)s when migrating from host %(src)s to host %(dest)s within the required time %(timeout)s.' % {'src': share['host'], 'dest': dest_host, 'share_id': share['id'], 'timeout': self.build_timeout, 'status': str(statuses)}
            raise tempest_lib_exc.TimeoutException(message)
    return share