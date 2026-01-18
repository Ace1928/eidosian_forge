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
def test_share_network_list_all_filters(self):
    filters = {'name': 'fake-name', 'project-id': '1234', 'created-since': '2001-01-01', 'created-before': '2002-02-02', 'neutron-net-id': 'fake-net', 'neutron-subnet-id': 'fake-subnet', 'network-type': 'local', 'segmentation-id': '5678', 'cidr': 'fake-cidr', 'ip-version': '4', 'offset': 10, 'limit': 20}
    command_str = 'share-network-list'
    for key, value in filters.items():
        command_str += ' --%(key)s=%(value)s' % {'key': key, 'value': value}
    self.run_command(command_str)
    query = utils.safe_urlencode(sorted([(k.replace('-', '_'), v) for k, v in filters.items()]))
    self.assert_called('GET', '/share-networks/detail?%s' % query)
    cliutils.print_list.assert_called_once_with(mock.ANY, fields=['id', 'name'])