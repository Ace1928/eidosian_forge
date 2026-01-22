import importlib
import os
from unittest import mock
from osc_lib.tests import utils as osc_lib_test_utils
import wrapt
from openstackclient import shell
def _assert_token_auth(self, cmd_options, default_args):
    with mock.patch(self.shell_class_name + '.initialize_app', self.app):
        _shell = osc_lib_test_utils.make_shell(shell_class=self.shell_class)
        _cmd = cmd_options + ' list role'
        osc_lib_test_utils.fake_execute(_shell, _cmd)
        self.app.assert_called_with(['list', 'role'])
        self.assertEqual(default_args.get('token', ''), _shell.options.token, 'token')
        self.assertEqual(default_args.get('auth_url', ''), _shell.options.auth_url, 'auth_url')