from unittest import mock
import uuid
import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_networks as osc_share_networks
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
@ddt.data(True, False)
def test_share_network_delete_with_wait(self, wait):
    oscutils.wait_for_delete = mock.Mock(return_value=True)
    share_networks = manila_fakes.FakeShareNetwork.create_share_networks(count=2)
    arglist = [share_networks[0].id, share_networks[1].name]
    if wait:
        arglist.append('--wait')
    verifylist = [('share_network', [share_networks[0].id, share_networks[1].name])]
    if wait:
        verifylist.append(('wait', True))
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    with mock.patch('osc_lib.utils.find_resource', side_effect=share_networks):
        result = self.cmd.take_action(parsed_args)
    self.assertEqual(self.share_networks_mock.delete.call_count, len(share_networks))
    if wait:
        oscutils.wait_for_delete.assert_has_calls([mock.call(manager=self.share_networks_mock, res_id=share_networks[0].id), mock.call(manager=self.share_networks_mock, res_id=share_networks[1].id)])
    self.assertIsNone(result)