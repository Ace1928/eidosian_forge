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
@ddt.data(*itertools.product(('--columns id,is_default', '--columns id,name', '--columns is_default', ''), {'2.45', '2.46', api_versions.MAX_VERSION}))
@ddt.unpack
def test_share_group_type_list(self, command_args, version):
    self.mock_object(shell_v2, '_print_share_group_type_list')
    command = 'share-group-type-list ' + command_args
    columns_requested = command_args.split('--columns ')[-1] or None
    is_default_in_api = api_versions.APIVersion(version) >= api_versions.APIVersion('2.46')
    self.run_command(command, version=version)
    if not is_default_in_api and (not columns_requested or 'is_default' in columns_requested):
        self.assert_called('GET', '/share-group-types/default')
        self.assert_called_anytime('GET', '/share-group-types')
    else:
        self.assert_called('GET', '/share-group-types')
    shell_v2._print_share_group_type_list.assert_called_once_with(mock.ANY, default_share_group_type=mock.ANY, columns=columns_requested)