from unittest import mock
from osc_lib import utils
from troveclient import common
from troveclient import exceptions
from troveclient.osc.v1 import database_configurations
from troveclient.tests.osc.v1 import fakes
class TestConfigurationInstancesList(TestConfigurations):
    defaults = {'limit': None, 'marker': None}
    columns = database_configurations.ListDatabaseConfigurationInstances.columns
    values = [('1', 'instance-1'), ('2', 'instance-2')]

    def setUp(self):
        super(TestConfigurationInstancesList, self).setUp()
        self.cmd = database_configurations.ListDatabaseConfigurationInstances(self.app, None)
        data = self.fake_configurations.get_configuration_instances()
        self.configuration_client.instances.return_value = common.Paginated(data)

    @mock.patch.object(utils, 'find_resource')
    def test_configuration_instances_list(self, mock_find):
        args = ['c-123']
        mock_find.return_value = args[0]
        parsed_args = self.check_parser(self.cmd, args, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.values, data)