import testscenarios
from unittest import mock
from cliff import command
from cliff import commandmanager
from cliff.tests import base
from cliff.tests import utils
class TestLegacyCommand(base.TestBase):

    def test_find_legacy(self):
        mgr = utils.TestCommandManager(None)
        mgr.add_command('new name', FauxCommand)
        mgr.add_legacy_command('old name', 'new name')
        cmd, name, remaining = mgr.find_command(['old', 'name'])
        self.assertIs(cmd, FauxCommand)
        self.assertEqual(name, 'old name')

    def test_legacy_overrides_new(self):
        mgr = utils.TestCommandManager(None)
        mgr.add_command('cmd1', FauxCommand)
        mgr.add_command('cmd2', FauxCommand2)
        mgr.add_legacy_command('cmd2', 'cmd1')
        cmd, name, remaining = mgr.find_command(['cmd2'])
        self.assertIs(cmd, FauxCommand)
        self.assertEqual(name, 'cmd2')

    def test_no_legacy(self):
        mgr = utils.TestCommandManager(None)
        mgr.add_command('cmd1', FauxCommand)
        self.assertRaises(ValueError, mgr.find_command, ['cmd2'])

    def test_no_command(self):
        mgr = utils.TestCommandManager(None)
        mgr.add_legacy_command('cmd2', 'cmd1')
        self.assertRaises(ValueError, mgr.find_command, ['cmd2'])