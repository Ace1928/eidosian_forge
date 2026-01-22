from unittest import mock
from osc_lib import exceptions
import testtools
from neutronclient.osc.v2.sfc import sfc_port_pair_group
from neutronclient.tests.unit.osc.v2.sfc import fakes
class TestCreateSfcPortPairGroup(fakes.TestNeutronClientOSCV2):
    _port_pair_group = fakes.FakeSfcPortPairGroup.create_port_pair_group()
    columns = ('Description', 'ID', 'Name', 'Port Pair', 'Port Pair Group Parameters', 'Project', 'Tap Enabled')

    def get_data(self, ppg):
        return (ppg['description'], ppg['id'], ppg['name'], ppg['port_pairs'], ppg['port_pair_group_parameters'], ppg['project_id'], ppg['tap_enabled'])

    def setUp(self):
        super(TestCreateSfcPortPairGroup, self).setUp()
        self.network.create_sfc_port_pair_group = mock.Mock(return_value=self._port_pair_group)
        self.data = self.get_data(self._port_pair_group)
        self.cmd = sfc_port_pair_group.CreateSfcPortPairGroup(self.app, self.namespace)

    def test_create_port_pair_group_default_options(self):
        arglist = ['--port-pair', self._port_pair_group['port_pairs'], self._port_pair_group['name']]
        verifylist = [('port_pairs', [self._port_pair_group['port_pairs']]), ('name', self._port_pair_group['name'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network.create_sfc_port_pair_group.assert_called_once_with(**{'name': self._port_pair_group['name'], 'port_pairs': [self._port_pair_group['port_pairs']]})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)

    def test_create_port_pair_group(self):
        arglist = ['--description', self._port_pair_group['description'], '--port-pair', self._port_pair_group['port_pairs'], self._port_pair_group['name']]
        verifylist = [('port_pairs', [self._port_pair_group['port_pairs']]), ('name', self._port_pair_group['name']), ('description', self._port_pair_group['description'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network.create_sfc_port_pair_group.assert_called_once_with(**{'name': self._port_pair_group['name'], 'port_pairs': [self._port_pair_group['port_pairs']], 'description': self._port_pair_group['description']})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)

    def test_create_tap_enabled_port_pair_group(self):
        arglist = ['--description', self._port_pair_group['description'], '--port-pair', self._port_pair_group['port_pairs'], self._port_pair_group['name'], '--enable-tap']
        verifylist = [('port_pairs', [self._port_pair_group['port_pairs']]), ('name', self._port_pair_group['name']), ('description', self._port_pair_group['description']), ('enable_tap', True)]
        expected_data = self._update_expected_response_data(data={'tap_enabled': True})
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network.create_sfc_port_pair_group.assert_called_once_with(**{'name': self._port_pair_group['name'], 'port_pairs': [self._port_pair_group['port_pairs']], 'description': self._port_pair_group['description'], 'tap_enabled': True})
        self.assertEqual(self.columns, columns)
        self.assertEqual(expected_data, data)

    def _update_expected_response_data(self, data):
        ppg = fakes.FakeSfcPortPairGroup.create_port_pair_group(data)
        self.network.create_sfc_port_pair_group.return_value = ppg
        return self.get_data(ppg)