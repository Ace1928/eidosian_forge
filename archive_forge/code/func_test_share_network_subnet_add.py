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
@ddt.data({}, {'--neutron_net_id': 'fake_neutron_net_id', '--neutron_subnet_id': 'fake_neutron_subnet_id'}, {'--availability-zone': 'fake_availability_zone_id'}, {'--neutron_net_id': 'fake_neutron_net_id', '--neutron_subnet_id': 'fake_neutron_subnet_id', '--availability-zone': 'fake_availability_zone_id'})
def test_share_network_subnet_add(self, data):
    fake_share_network = type('FakeShareNetwork', (object,), {'id': '1234'})
    self.mock_object(shell_v2, '_find_share_network', mock.Mock(return_value=fake_share_network))
    cmd = 'share-network-subnet-create'
    for k, v in data.items():
        cmd += ' ' + k + ' ' + v
    cmd += ' ' + fake_share_network.id
    self.run_command(cmd)
    shell_v2._find_share_network.assert_called_once_with(mock.ANY, fake_share_network.id)
    self.assert_called('POST', '/share-networks/1234/subnets')