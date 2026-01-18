from unittest import mock
from osc_lib import exceptions
import testtools
from neutronclient.osc.v2.sfc import sfc_port_chain
from neutronclient.tests.unit.osc.v2.sfc import fakes
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