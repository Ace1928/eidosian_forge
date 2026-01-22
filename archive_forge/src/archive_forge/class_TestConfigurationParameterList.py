from unittest import mock
from osc_lib import utils
from troveclient import common
from troveclient import exceptions
from troveclient.osc.v1 import database_configurations
from troveclient.tests.osc.v1 import fakes
class TestConfigurationParameterList(TestConfigurations):
    columns = database_configurations.ListDatabaseConfigurationParameters.columns
    values = ('connect_timeout', 'integer', 2, 31536000, 'false')

    def setUp(self):
        super(TestConfigurationParameterList, self).setUp()
        self.cmd = database_configurations.ListDatabaseConfigurationParameters(self.app, None)
        data = [self.fake_configuration_params.get_params_connect_timeout()]
        self.configuration_params_client.parameters.return_value = common.Paginated(data)
        self.configuration_params_client.parameters_by_version.return_value = common.Paginated(data)

    def test_configuration_parameters_list_defaults(self):
        args = ['d-123', '--datastore', 'mysql']
        verifylist = [('datastore_version', 'd-123'), ('datastore', 'mysql')]
        parsed_args = self.check_parser(self.cmd, args, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.columns, columns)
        self.assertEqual([tuple(self.values)], data)

    def test_configuration_parameters_list_with_version_id_exception(self):
        args = ['d-123']
        verifylist = [('datastore_version', 'd-123')]
        parsed_args = self.check_parser(self.cmd, args, verifylist)
        self.assertRaises(exceptions.NoUniqueMatch, self.cmd.take_action, parsed_args)