import argparse
from unittest import mock
import uuid
from keystoneclient.auth.identity.generic import cli
from keystoneclient import exceptions
from keystoneclient.tests.unit import utils
@mock.patch('sys.stdin', autospec=True)
@mock.patch('getpass.getpass')
def test_prompt_no_password(self, mock_getpass, mock_stdin):
    mock_stdin.isatty = lambda: True
    mock_getpass.return_value = ''
    exc = self.assertRaises(exceptions.CommandError, self.new_plugin, ['--os-auth-url', uuid.uuid4().hex, '--os-username', uuid.uuid4().hex])
    self.assertIn('password', str(exc))