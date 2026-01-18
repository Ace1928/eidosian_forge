import copy
from unittest import mock
import fixtures
from osc_lib.tests import utils as osc_lib_utils
from openstackclient import shell
from openstackclient.tests.unit.integ import base as test_base
from openstackclient.tests.unit import test_shell
def test_shell_args_options(self):
    """Verify command line options override environment variables"""
    _shell = shell.OpenStackShell()
    _shell.run('--os-username zarquon --os-password qaz extension list'.split())
    self.assertNotEqual(len(self.requests_mock.request_history), 0)
    self.assertEqual(test_base.V3_AUTH_URL, self.requests_mock.request_history[0].url)
    auth_req = self.requests_mock.request_history[1].json()
    self.assertEqual('qaz', auth_req['auth']['identity']['password']['user']['password'])
    self.assertEqual(test_shell.DEFAULT_PROJECT_DOMAIN_ID, auth_req['auth']['identity']['password']['user']['domain']['id'])
    self.assertEqual('zarquon', auth_req['auth']['identity']['password']['user']['name'])