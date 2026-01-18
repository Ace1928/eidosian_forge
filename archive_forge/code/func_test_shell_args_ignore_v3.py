import copy
from unittest import mock
import fixtures
from osc_lib.tests import utils as osc_lib_utils
from openstackclient import shell
from openstackclient.tests.unit.integ import base as test_base
from openstackclient.tests.unit import test_shell
def test_shell_args_ignore_v3(self):
    _shell = shell.OpenStackShell()
    _shell.run('extension list'.split())
    self.assertNotEqual(len(self.requests_mock.request_history), 0)
    self.assertEqual(test_base.V2_AUTH_URL, self.requests_mock.request_history[0].url)
    auth_req = self.requests_mock.request_history[1].json()
    self.assertEqual(test_shell.DEFAULT_PROJECT_NAME, auth_req['auth']['tenantName'])
    self.assertEqual(test_shell.DEFAULT_USERNAME, auth_req['auth']['passwordCredentials']['username'])
    self.assertEqual(test_shell.DEFAULT_PASSWORD, auth_req['auth']['passwordCredentials']['password'])