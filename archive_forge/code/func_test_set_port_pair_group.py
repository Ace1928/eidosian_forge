from unittest import mock
from osc_lib import exceptions
import testtools
from neutronclient.osc.v2.sfc import sfc_port_pair_group
from neutronclient.tests.unit.osc.v2.sfc import fakes
def test_set_port_pair_group(self):
    target = self.resource['id']
    port_pair1 = 'additional_port1'
    port_pair2 = 'additional_port2'
    self.network.find_sfc_port_pair = mock.Mock(side_effect=lambda name_or_id, ignore_missing=False: {'id': name_or_id})
    self.network.find_sfc_port_pair_group = mock.Mock(side_effect=lambda name_or_id, ignore_missing=False: {'id': name_or_id, 'port_pairs': self.ppg_pp})
    arglist = [target, '--port-pair', port_pair1, '--port-pair', port_pair2]
    verifylist = [(self.res, target), ('port_pairs', [port_pair1, port_pair2])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    expect = {'port_pairs': sorted([*self.ppg_pp, port_pair1, port_pair2])}
    self.mocked.assert_called_once_with(target, **expect)
    self.assertIsNone(result)