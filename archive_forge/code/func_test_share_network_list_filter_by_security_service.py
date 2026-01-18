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
@mock.patch.object(cliutils, 'print_list', mock.Mock())
@mock.patch.object(shell_v2, '_find_security_service', mock.Mock())
def test_share_network_list_filter_by_security_service(self):
    ss = type('FakeSecurityService', (object,), {'id': 'fake-ss-id'})
    shell_v2._find_security_service.return_value = ss
    for command in ['--security_service', '--security-service']:
        self.run_command('share-network-list %(command)s %(ss_id)s' % {'command': command, 'ss_id': ss.id})
        self.assert_called('GET', '/share-networks/detail?security_service_id=%s' % ss.id)
        shell_v2._find_security_service.assert_called_with(mock.ANY, ss.id)
        cliutils.print_list.assert_called_with(mock.ANY, fields=['id', 'name'])