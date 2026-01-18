from unittest import mock
from oslo_serialization import jsonutils as json
from saharaclient.api import plugins as api_plugins
from saharaclient.osc.v1 import plugins as osc_plugins
from saharaclient.tests.unit.osc.v1 import fakes
def test_plugin_version_show(self):
    arglist = ['fake', '--plugin-version', '0.1']
    verifylist = [('plugin', 'fake'), ('plugin_version', '0.1')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.plugins_mock.get_version_details.assert_called_once_with('fake', '0.1')
    expected_columns = ('Description', 'Name', 'Required image tags', 'Title', '', 'Plugin version 0.1: enabled', 'Plugin: enabled', '', 'Service:', '', 'HDFS', 'MapReduce')
    self.assertEqual(expected_columns, columns)
    expected_data = ('Plugin for tests', 'fake', '0.1, fake', 'Fake Plugin', '', True, True, '', 'Available processes:', '', 'datanode, namenode', 'jobtracker, tasktracker')
    self.assertEqual(expected_data, data)