from unittest import mock
from osc_lib import exceptions
import testtools
from neutronclient.osc.v2.sfc import sfc_port_chain
from neutronclient.tests.unit.osc.v2.sfc import fakes
class TestSetSfcPortChain(fakes.TestNeutronClientOSCV2):
    _port_chain = fakes.FakeSfcPortChain.create_port_chain()
    resource = _port_chain
    res = 'port_chain'
    _port_chain_name = _port_chain['name']
    _port_chain_id = _port_chain['id']
    pc_ppg = _port_chain['port_pair_groups']
    pc_fc = _port_chain['flow_classifiers']

    def setUp(self):
        super(TestSetSfcPortChain, self).setUp()
        self.mocked = self.network.update_sfc_port_chain
        self.cmd = sfc_port_chain.SetSfcPortChain(self.app, self.namespace)

    def test_set_port_chain(self):
        client = self.app.client_manager.network
        mock_port_chain_update = client.update_sfc_port_chain
        arglist = [self._port_chain_name, '--name', 'name_updated', '--description', 'desc_updated']
        verifylist = [('port_chain', self._port_chain_name), ('name', 'name_updated'), ('description', 'desc_updated')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'name': 'name_updated', 'description': 'desc_updated'}
        mock_port_chain_update.assert_called_once_with(self._port_chain_name, **attrs)
        self.assertIsNone(result)

    def test_set_flow_classifiers(self):
        target = self.resource['id']
        fc1 = 'flow_classifier1'
        fc2 = 'flow_classifier2'
        self.network.find_sfc_port_chain = mock.Mock(side_effect=lambda name_or_id, ignore_missing=False: {'id': name_or_id, 'flow_classifiers': [self.pc_fc]})
        self.network.find_sfc_flow_classifier.side_effect = lambda name_or_id, ignore_missing=False: {'id': name_or_id}
        arglist = [target, '--flow-classifier', fc1, '--flow-classifier', fc2]
        verifylist = [(self.res, target), ('flow_classifiers', [fc1, fc2])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        expect = {'flow_classifiers': [self.pc_fc, fc1, fc2]}
        self.mocked.assert_called_once_with(target, **expect)
        self.assertIsNone(result)

    def test_set_no_flow_classifier(self):
        client = self.app.client_manager.network
        mock_port_chain_update = client.update_sfc_port_chain
        arglist = [self._port_chain_name, '--no-flow-classifier']
        verifylist = [('port_chain', self._port_chain_name), ('no_flow_classifier', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'flow_classifiers': []}
        mock_port_chain_update.assert_called_once_with(self._port_chain_name, **attrs)
        self.assertIsNone(result)

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

    def test_set_no_port_pair_group(self):
        target = self.resource['id']
        ppg1 = 'port_pair_group1'
        arglist = [target, '--no-port-pair-group', '--port-pair-group', ppg1]
        verifylist = [(self.res, target), ('no_port_pair_group', True), ('port_pair_groups', [ppg1])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        expect = {'port_pair_groups': [ppg1]}
        self.mocked.assert_called_once_with(target, **expect)
        self.assertIsNone(result)

    def test_set_only_no_port_pair_group(self):
        target = self.resource['id']
        arglist = [target, '--no-port-pair-group']
        verifylist = [(self.res, target), ('no_port_pair_group', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)