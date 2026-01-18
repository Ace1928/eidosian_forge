from unittest import mock
from osc_lib import exceptions
import testtools
from neutronclient.osc.v2.sfc import sfc_port_chain
from neutronclient.tests.unit.osc.v2.sfc import fakes
def test_set_port_pair_groups(self):
    target = self.resource['id']
    existing_ppg = self.pc_ppg
    ppg1 = 'port_pair_group1'
    ppg2 = 'port_pair_group2'
    self.network.find_sfc_port_chain = mock.Mock(side_effect=lambda name_or_id, ignore_missing=False: {'id': name_or_id, 'port_pair_groups': [self.pc_ppg]})
    arglist = [target, '--port-pair-group', ppg1, '--port-pair-group', ppg2]
    verifylist = [(self.res, target), ('port_pair_groups', [ppg1, ppg2])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    expect = {'port_pair_groups': [existing_ppg, ppg1, ppg2]}
    self.mocked.assert_called_once_with(target, **expect)
    self.assertIsNone(result)