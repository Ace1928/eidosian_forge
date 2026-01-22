import testscenarios
from unittest import mock
from cliff import command
from cliff import commandmanager
from cliff.tests import base
from cliff.tests import utils
class TestCommandManagerGroups(base.TestBase):

    def test_add_command_group(self):
        mgr = FakeCommandManager('test')
        mock_cmd_one = mock.Mock()
        mgr.add_command('mock', mock_cmd_one)
        cmd_mock, name, args = mgr.find_command(['mock'])
        self.assertEqual(mock_cmd_one, cmd_mock)
        cmd_one, name, args = mgr.find_command(['one'])
        self.assertEqual(FAKE_CMD_ONE, cmd_one)
        mgr.add_command_group('greek')
        cmd_alpha, name, args = mgr.find_command(['alpha'])
        self.assertEqual(FAKE_CMD_ALPHA, cmd_alpha)
        cmd_two, name, args = mgr.find_command(['two'])
        self.assertEqual(FAKE_CMD_TWO, cmd_two)

    def test_get_command_groups(self):
        mgr = FakeCommandManager('test')
        mock_cmd_one = mock.Mock()
        mgr.add_command('mock', mock_cmd_one)
        cmd_mock, name, args = mgr.find_command(['mock'])
        self.assertEqual(mock_cmd_one, cmd_mock)
        mgr.add_command_group('greek')
        gl = mgr.get_command_groups()
        self.assertEqual(['test', 'greek'], gl)

    def test_get_command_names(self):
        mock_cmd_one = mock.Mock()
        mock_cmd_one.name = 'one'
        mock_cmd_two = mock.Mock()
        mock_cmd_two.name = 'cmd two'
        mock_get_group_all = mock.Mock(return_value=[mock_cmd_one, mock_cmd_two])
        with mock.patch('stevedore.ExtensionManager', mock_get_group_all) as mock_manager:
            mgr = commandmanager.CommandManager('test')
            mock_manager.assert_called_once_with('test')
            cmds = mgr.get_command_names('test')
            self.assertEqual(['one', 'cmd two'], cmds)