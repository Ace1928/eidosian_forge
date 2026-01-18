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
@ddt.data('--wait', '')
def test_shrink_with_wait_option(self, wait_option):
    available_share = shares.Share('fake', {'id': '1234', 'status': 'available'})
    share_to_shrink = shares.Share('fake', {'id': '1234', 'status': 'shrinking'})
    fake_shares = [available_share, share_to_shrink, share_to_shrink, available_share]
    self.mock_object(shell_v2, '_find_share', mock.Mock(side_effect=fake_shares))
    expected_shrink_body = {'shrink': {'new_size': 77}}
    self.run_command('shrink 1234 77 %s' % wait_option)
    self.assert_called_anytime('POST', '/shares/1234/action', body=expected_shrink_body, clear_callstack=False)
    if wait_option:
        shell_v2._find_share.assert_has_calls([mock.call(self.shell.cs, '1234')] * 4)
        self.assertEqual(4, shell_v2._find_share.call_count)
    else:
        shell_v2._find_share.assert_called_with(self.shell.cs, '1234')
        self.assertEqual(2, shell_v2._find_share.call_count)