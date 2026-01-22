import argparse
import codecs
import io
from unittest import mock
from cliff import app as application
from cliff import command as c_cmd
from cliff import commandmanager
from cliff.tests import base
from cliff.tests import utils as test_utils
from cliff import utils
import sys
class TestOptionParser(base.TestBase):

    def test_conflicting_option_should_throw(self):

        class MyApp(application.App):

            def __init__(self):
                super(MyApp, self).__init__(description='testing', version='0.1', command_manager=commandmanager.CommandManager('tests'))

            def build_option_parser(self, description, version):
                parser = super(MyApp, self).build_option_parser(description, version)
                parser.add_argument('-h', '--help', default=self, help='Show help message and exit.')
        self.assertRaises(argparse.ArgumentError, MyApp)

    def test_conflicting_option_custom_arguments_should_not_throw(self):

        class MyApp(application.App):

            def __init__(self):
                super(MyApp, self).__init__(description='testing', version='0.1', command_manager=commandmanager.CommandManager('tests'))

            def build_option_parser(self, description, version):
                argparse_kwargs = {'conflict_handler': 'resolve'}
                parser = super(MyApp, self).build_option_parser(description, version, argparse_kwargs=argparse_kwargs)
                parser.add_argument('-h', '--help', default=self, help='Show help message and exit.')
        MyApp()

    def test_option_parser_abbrev_issue(self):

        class MyCommand(c_cmd.Command):

            def get_parser(self, prog_name):
                parser = super(MyCommand, self).get_parser(prog_name)
                parser.add_argument('--end')
                return parser

            def take_action(self, parsed_args):
                assert parsed_args.end == '123'

        class MyCommandManager(commandmanager.CommandManager):

            def load_commands(self, namespace):
                self.add_command('mycommand', MyCommand)

        class MyApp(application.App):

            def __init__(self):
                super(MyApp, self).__init__(description='testing', version='0.1', command_manager=MyCommandManager(None))

            def build_option_parser(self, description, version):
                parser = super(MyApp, self).build_option_parser(description, version, argparse_kwargs={'allow_abbrev': False})
                parser.add_argument('--endpoint')
                return parser
        app = MyApp()
        app.run(['--debug', 'mycommand', '--end', '123'])