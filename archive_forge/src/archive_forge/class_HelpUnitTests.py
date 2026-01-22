from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from gslib.command import Command
import gslib.tests.testcase as testcase
class HelpUnitTests(testcase.GsUtilUnitTestCase):
    """Help command unit test suite."""

    def test_help_noargs(self):
        stdout = self.RunCommand('help', return_stdout=True)
        self.assertIn('Available commands', stdout)

    def test_help_subcommand_arg(self):
        stdout = self.RunCommand('help', ['web', 'set'], return_stdout=True)
        self.assertIn('gsutil web set', stdout)
        self.assertNotIn('gsutil web get', stdout)

    def test_help_invalid_subcommand_arg(self):
        stdout = self.RunCommand('help', ['web', 'asdf'], return_stdout=True)
        self.assertIn('help about one of the subcommands', stdout)

    def test_help_with_subcommand_for_command_without_subcommands(self):
        stdout = self.RunCommand('help', ['ls', 'asdf'], return_stdout=True)
        self.assertIn('has no subcommands', stdout)

    def test_help_command_arg(self):
        stdout = self.RunCommand('help', ['ls'], return_stdout=True)
        self.assertIn('ls - List providers, buckets', stdout)

    def test_command_help_arg(self):
        stdout = self.RunCommand('ls', ['--help'], return_stdout=True)
        self.assertIn('ls - List providers, buckets', stdout)

    def test_subcommand_help_arg(self):
        stdout = self.RunCommand('web', ['set', '--help'], return_stdout=True)
        self.assertIn('gsutil web set', stdout)
        self.assertNotIn('gsutil web get', stdout)

    def test_command_args_with_help(self):
        stdout = self.RunCommand('cp', ['foo', 'bar', '--help'], return_stdout=True)
        self.assertIn('cp - Copy files and objects', stdout)