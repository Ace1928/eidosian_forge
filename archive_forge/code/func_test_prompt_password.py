import argparse
import copy
from unittest import mock
import uuid
from keystoneauth1 import fixture
from keystoneclient import access
from keystoneclient.auth.identity import v3
from keystoneclient.auth.identity.v3 import base as v3_base
from keystoneclient import client
from keystoneclient import exceptions
from keystoneclient import session
from keystoneclient.tests.unit import utils
@mock.patch('sys.stdin', autospec=True)
def test_prompt_password(self, mock_stdin):
    parser = argparse.ArgumentParser()
    v3.Password.register_argparse_arguments(parser)
    username = uuid.uuid4().hex
    user_domain_id = uuid.uuid4().hex
    auth_url = uuid.uuid4().hex
    project_id = uuid.uuid4().hex
    password = uuid.uuid4().hex
    opts = parser.parse_args(['--os-username', username, '--os-auth-url', auth_url, '--os-user-domain-id', user_domain_id, '--os-project-id', project_id])
    with mock.patch('getpass.getpass') as mock_getpass:
        mock_getpass.return_value = password
        mock_stdin.isatty = lambda: True
        plugin = v3.Password.load_from_argparse_arguments(opts)
        self.assertEqual(auth_url, plugin.auth_url)
        self.assertEqual(username, plugin.auth_methods[0].username)
        self.assertEqual(project_id, plugin.project_id)
        self.assertEqual(user_domain_id, plugin.auth_methods[0].user_domain_id)
        self.assertEqual(password, plugin.auth_methods[0].password)