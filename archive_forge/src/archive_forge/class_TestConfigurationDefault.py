from unittest import mock
from osc_lib import utils
from troveclient import common
from troveclient import exceptions
from troveclient.osc.v1 import database_configurations
from troveclient.tests.osc.v1 import fakes
class TestConfigurationDefault(TestConfigurations):
    values = ('2', '98', '1', '15M')

    def setUp(self):
        super(TestConfigurationDefault, self).setUp()
        self.cmd = database_configurations.DefaultDatabaseConfiguration(self.app, None)
        self.data = self.fake_configurations.get_default_configuration()
        self.instance_client.configuration.return_value = self.data
        self.columns = ('innodb_log_files_in_group', 'max_user_connections', 'skip-external-locking', 'tmp_table_size')

    @mock.patch.object(utils, 'find_resource')
    def test_default_database_configuration(self, mock_find):
        args = ['1234']
        mock_find.return_value = args[0]
        parsed_args = self.check_parser(self.cmd, args, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.values, data)