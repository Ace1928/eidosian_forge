import testscenarios
from unittest import mock
from cliff import command
from cliff import commandmanager
from cliff.tests import base
from cliff.tests import utils
class TestLoad(base.TestBase):

    def test_load_commands(self):
        testcmd = mock.Mock(name='testcmd')
        testcmd.name.replace.return_value = 'test'
        mock_get_group_all = mock.Mock(return_value=[testcmd])
        with mock.patch('stevedore.ExtensionManager', mock_get_group_all) as mock_manager:
            mgr = commandmanager.CommandManager('test')
            mock_manager.assert_called_once_with('test')
            names = [n for n, v in mgr]
            self.assertEqual(['test'], names)

    def test_load_commands_keep_underscores(self):
        testcmd = mock.Mock()
        testcmd.name = 'test_cmd'
        mock_get_group_all = mock.Mock(return_value=[testcmd])
        with mock.patch('stevedore.ExtensionManager', mock_get_group_all) as mock_manager:
            mgr = commandmanager.CommandManager('test', convert_underscores=False)
            mock_manager.assert_called_once_with('test')
            names = [n for n, v in mgr]
            self.assertEqual(['test_cmd'], names)

    def test_load_commands_replace_underscores(self):
        testcmd = mock.Mock()
        testcmd.name = 'test_cmd'
        mock_get_group_all = mock.Mock(return_value=[testcmd])
        with mock.patch('stevedore.ExtensionManager', mock_get_group_all) as mock_manager:
            mgr = commandmanager.CommandManager('test', convert_underscores=True)
            mock_manager.assert_called_once_with('test')
            names = [n for n, v in mgr]
            self.assertEqual(['test cmd'], names)