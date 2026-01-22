from unittest import mock
from osc_lib import utils
from oslo_utils import uuidutils
from troveclient import common
from troveclient import exceptions
from troveclient.osc.v1 import database_instances
from troveclient.tests.osc.v1 import fakes
from troveclient.v1 import instances
class TestDatabaseInstanceUpgrade(TestInstances):

    def setUp(self):
        super(TestDatabaseInstanceUpgrade, self).setUp()
        self.cmd = database_instances.UpgradeDatabaseInstance(self.app, None)

    @mock.patch.object(utils, 'find_resource')
    def test_instance_upgrade(self, mock_find):
        args = ['instance1', 'datastore_version1']
        mock_find.return_value = args[0]
        parsed_args = self.check_parser(self.cmd, args, [])
        result = self.cmd.take_action(parsed_args)
        self.instance_client.upgrade.assert_called_with('instance1', 'datastore_version1')
        self.assertIsNone(result)