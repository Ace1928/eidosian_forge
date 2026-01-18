from unittest import mock
from oslo_serialization import jsonutils as json
from saharaclient.api import plugins as api_plugins
from saharaclient.osc.v1 import plugins as osc_plugins
from saharaclient.tests.unit.osc.v1 import fakes
def test_plugin_show(self):
    arglist = ['fake']
    verifylist = [('plugin', 'fake')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.plugins_mock.get.assert_called_once_with('fake')
    expected_columns = ('Description', 'Name', 'Title', 'Versions', '', 'Plugin version 0.1: enabled', 'Plugin: enabled')
    self.assertEqual(expected_columns, columns)
    expected_data = ('Plugin for tests', 'fake', 'Fake Plugin', '0.1, 0.2', '', True, True)
    self.assertEqual(expected_data, data)