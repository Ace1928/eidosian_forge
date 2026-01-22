from unittest import mock
from osc_lib import utils
from troveclient import common
from troveclient import exceptions
from troveclient.osc.v1 import database_configurations
from troveclient.tests.osc.v1 import fakes
class TestConfigurationParameterShow(TestConfigurations):
    values = ('d-123', 31536000, 2, 'connect_timeout', 'false', 'integer')

    def setUp(self):
        super(TestConfigurationParameterShow, self).setUp()
        self.cmd = database_configurations.ShowDatabaseConfigurationParameter(self.app, None)
        data = self.fake_configuration_params.get_params_connect_timeout()
        self.configuration_params_client.get_parameter.return_value = data
        self.configuration_params_client.get_parameter_by_version.return_value = data
        self.columns = ('datastore_version_id', 'max', 'min', 'name', 'restart_required', 'type')

    def test_configuration_parameter_show_defaults(self):
        args = ['d-123', 'connect_timeout', '--datastore', 'mysql']
        verifylist = [('datastore_version', 'd-123'), ('parameter', 'connect_timeout'), ('datastore', 'mysql')]
        parsed_args = self.check_parser(self.cmd, args, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.values, data)

    def test_configuration_parameter_show_with_version_id_exception(self):
        args = ['d-123', 'connect_timeout']
        verifylist = [('datastore_version', 'd-123'), ('parameter', 'connect_timeout')]
        parsed_args = self.check_parser(self.cmd, args, verifylist)
        self.assertRaises(exceptions.NoUniqueMatch, self.cmd.take_action, parsed_args)