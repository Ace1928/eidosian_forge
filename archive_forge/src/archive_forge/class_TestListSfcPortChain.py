from unittest import mock
from osc_lib import exceptions
import testtools
from neutronclient.osc.v2.sfc import sfc_port_chain
from neutronclient.tests.unit.osc.v2.sfc import fakes
class TestListSfcPortChain(fakes.TestNeutronClientOSCV2):
    _port_chains = fakes.FakeSfcPortChain.create_port_chains(count=1)
    columns = ('ID', 'Name', 'Port Pair Groups', 'Flow Classifiers', 'Chain Parameters')
    columns_long = ('ID', 'Name', 'Port Pair Groups', 'Flow Classifiers', 'Chain Parameters', 'Description', 'Project')
    _port_chain = _port_chains[0]
    data = [_port_chain['id'], _port_chain['name'], _port_chain['port_pair_groups'], _port_chain['flow_classifiers'], _port_chain['chain_parameters']]
    data_long = [_port_chain['id'], _port_chain['name'], _port_chain['project_id'], _port_chain['port_pair_groups'], _port_chain['flow_classifiers'], _port_chain['chain_parameters'], _port_chain['description']]
    _port_chain1 = {'port_chains': _port_chain}
    _port_chain_id = _port_chain['id']

    def setUp(self):
        super(TestListSfcPortChain, self).setUp()
        self.network.sfc_port_chains = mock.Mock(return_value=self._port_chains)
        self.cmd = sfc_port_chain.ListSfcPortChain(self.app, self.namespace)

    def test_list_port_chains(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns = self.cmd.take_action(parsed_args)[0]
        pcs = self.network.sfc_port_chains()
        pc = pcs[0]
        data = [pc['id'], pc['name'], pc['port_pair_groups'], pc['flow_classifiers'], pc['chain_parameters']]
        self.assertEqual(list(self.columns), columns)
        self.assertEqual(self.data, data)

    def test_list_port_chain_with_long_opion(self):
        arglist = ['--long']
        verifylist = [('long', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns = self.cmd.take_action(parsed_args)[0]
        pcs = self.network.sfc_port_chains()
        pc = pcs[0]
        data = [pc['id'], pc['name'], pc['project_id'], pc['port_pair_groups'], pc['flow_classifiers'], pc['chain_parameters'], pc['description']]
        self.assertEqual(list(self.columns_long), columns)
        self.assertEqual(self.data_long, data)