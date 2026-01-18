import io
import os
import sys
from unittest import mock
from cliff import app as application
from cliff import commandmanager
from cliff import help
from cliff.tests import base
from cliff.tests import utils
def test_show_help_for_help(self):
    stdout = io.StringIO()
    app = application.App('testing', '1', utils.TestCommandManager(utils.TEST_NAMESPACE), stdout=stdout)
    app.NAME = 'test'
    app.options = mock.Mock()
    help_cmd = help.HelpCommand(app, mock.Mock())
    parser = help_cmd.get_parser('test')
    parsed_args = parser.parse_args([])
    try:
        help_cmd.run(parsed_args)
    except help.HelpExit:
        pass
    help_text = stdout.getvalue()
    basecommand = os.path.split(sys.argv[0])[1]
    self.assertIn('usage: %s [--version]' % basecommand, help_text)
    self.assertRegex(help_text, 'option(s|al arguments):\n  --version')
    expected = '  one            Test command\n  three word command  Test command\n'
    self.assertIn(expected, help_text)