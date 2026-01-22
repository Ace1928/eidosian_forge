from unittest import mock
from osc_lib import exceptions
import testtools
from neutronclient.osc.v2.sfc import sfc_port_chain
from neutronclient.tests.unit.osc.v2.sfc import fakes
class TestShowSfcPortChain(fakes.TestNeutronClientOSCV2):
    _pc = fakes.FakeSfcPortChain.create_port_chain()
    data = (_pc['chain_parameters'], _pc['description'], _pc['flow_classifiers'], _pc['id'], _pc['name'], _pc['port_pair_groups'], _pc['project_id'])
    _port_chain = _pc
    _port_chain_id = _pc['id']
    columns = ('Chain Parameters', 'Description', 'Flow Classifiers', 'ID', 'Name', 'Port Pair Groups', 'Project')

    def setUp(self):
        super(TestShowSfcPortChain, self).setUp()
        self.network.get_sfc_port_chain = mock.Mock(return_value=self._port_chain)
        self.cmd = sfc_port_chain.ShowSfcPortChain(self.app, self.namespace)

    def test_show_port_chain(self):
        client = self.app.client_manager.network
        mock_port_chain_show = client.get_sfc_port_chain
        arglist = [self._port_chain_id]
        verifylist = [('port_chain', self._port_chain_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        mock_port_chain_show.assert_called_once_with(self._port_chain_id)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)