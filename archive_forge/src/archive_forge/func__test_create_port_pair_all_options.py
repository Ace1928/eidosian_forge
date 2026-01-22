from unittest import mock
from osc_lib import exceptions
import testtools
from neutronclient.osc.v2.sfc import sfc_port_pair
from neutronclient.tests.unit.osc.v2.sfc import fakes
def _test_create_port_pair_all_options(self, correlation):
    arglist = ['--description', self._port_pair['description'], '--egress', self._port_pair['egress'], '--ingress', self._port_pair['ingress'], self._port_pair['name'], '--service-function-parameters', 'correlation=%s,weight=1' % correlation]
    verifylist = [('ingress', self._port_pair['ingress']), ('egress', self._port_pair['egress']), ('name', self._port_pair['name']), ('description', self._port_pair['description']), ('service_function_parameters', [{'correlation': correlation, 'weight': '1'}])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    if correlation == 'None':
        correlation_param = None
    else:
        correlation_param = correlation
    self.network.create_sfc_port_pair.assert_called_once_with(**{'name': self._port_pair['name'], 'ingress': self._port_pair['ingress'], 'egress': self._port_pair['egress'], 'description': self._port_pair['description'], 'service_function_parameters': {'correlation': correlation_param, 'weight': '1'}})
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, data)