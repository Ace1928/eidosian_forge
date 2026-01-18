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
def wait_for_share_replica_status(self, share_replica, status='available', microversion=None):
    """Waits for a share replica to reach a given status."""
    replica = self.get_share_replica(share_replica, microversion=microversion)
    share_replica_status = replica['status']
    start = int(time.time())
    while share_replica_status != status:
        time.sleep(self.build_interval)
        replica = self.get_share_replica(share_replica, microversion=microversion)
        share_replica_status = replica['status']
        if share_replica_status == status:
            return replica
        elif 'error' in share_replica_status.lower():
            raise exceptions.ShareReplicaBuildErrorException(replica=share_replica)
        if int(time.time()) - start >= self.build_timeout:
            message = 'Share replica %(id)s failed to reach %(status)s status within the required time (%(build_timeout)s s).' % {'id': share_replica, 'status': status, 'build_timeout': self.build_timeout}
            raise tempest_lib_exc.TimeoutException(message)
    return replica