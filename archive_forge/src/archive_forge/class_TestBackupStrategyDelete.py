from troveclient.osc.v1 import database_backup_strategy
from troveclient.tests.osc.v1 import fakes
from troveclient.v1 import backup_strategy
class TestBackupStrategyDelete(TestBackupStrategy):

    def setUp(self):
        super(TestBackupStrategyDelete, self).setUp()
        self.cmd = database_backup_strategy.DeleteDatabaseBackupStrategy(self.app, None)

    def test_delete(self):
        args = ['--instance-id', 'fake_instance_id', '--project-id', 'fake_project']
        parsed_args = self.check_parser(self.cmd, args, [])
        self.cmd.take_action(parsed_args)
        self.manager.delete.assert_called_once_with(project_id='fake_project', instance_id='fake_instance_id')