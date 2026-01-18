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
@ddt.data('migration-start', 'migration-check')
def test_share_server_migration_start_and_check(self, method):
    command = 'share-server-%s 1234 host@backend --new-share-network 1111 --writable False --nondisruptive True --preserve-snapshots True' % method
    self.run_command(command)
    method = method.replace('-', '_')
    expected = {method: {'host': 'host@backend', 'writable': 'False', 'nondisruptive': 'True', 'preserve_snapshots': 'True', 'new_share_network_id': 1111}}
    self.assert_called('POST', '/share-servers/1234/action', body=expected)