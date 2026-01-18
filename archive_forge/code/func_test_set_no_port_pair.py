from unittest import mock
from osc_lib import exceptions
import testtools
from neutronclient.osc.v2.sfc import sfc_port_pair_group
from neutronclient.tests.unit.osc.v2.sfc import fakes
def test_set_no_port_pair(self):
    client = self.app.client_manager.network
    mock_port_pair_group_update = client.update_sfc_port_pair_group
    arglist = [self._port_pair_group_name, '--name', 'name_updated', '--description', 'desc_updated', '--no-port-pair']
    verifylist = [('port_pair_group', self._port_pair_group_name), ('name', 'name_updated'), ('description', 'desc_updated'), ('no_port_pair', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    attrs = {'name': 'name_updated', 'description': 'desc_updated', 'port_pairs': []}
    mock_port_pair_group_update.assert_called_once_with(self._port_pair_group_name, **attrs)
    self.assertIsNone(result)