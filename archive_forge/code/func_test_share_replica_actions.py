import itertools
from unittest import mock
import ddt
import fixtures
from oslo_utils import strutils
from manilaclient import api_versions
from manilaclient import client
from manilaclient.common.apiclient import utils as apiclient_utils
from manilaclient.common import cliutils
from manilaclient.common import constants
from manilaclient import exceptions
from manilaclient import shell
from manilaclient.tests.unit import utils as test_utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient import utils
from manilaclient.v2 import messages
from manilaclient.v2 import security_services
from manilaclient.v2 import share_access_rules
from manilaclient.v2 import share_group_types
from manilaclient.v2 import share_groups
from manilaclient.v2 import share_instances
from manilaclient.v2 import share_network_subnets
from manilaclient.v2 import share_networks
from manilaclient.v2 import share_servers
from manilaclient.v2 import share_snapshots
from manilaclient.v2 import share_types
from manilaclient.v2 import shares
from manilaclient.v2 import shell as shell_v2
@ddt.data('promote', 'resync')
@mock.patch.object(shell_v2, '_find_share_replica', mock.Mock())
def test_share_replica_actions(self, action):
    fake_replica = type('FakeShareReplica', (object,), {'id': '1234'})
    shell_v2._find_share_replica.return_value = fake_replica
    cmd = 'share-replica-' + action + ' ' + fake_replica.id
    self.run_command(cmd)
    self.assert_called('POST', '/share-replicas/1234/action', body={action.replace('-', '_'): None})