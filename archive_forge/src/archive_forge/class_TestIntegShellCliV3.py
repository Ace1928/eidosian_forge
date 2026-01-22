import copy
from unittest import mock
import fixtures
from osc_lib.tests import utils as osc_lib_utils
from openstackclient import shell
from openstackclient.tests.unit.integ import base as test_base
from openstackclient.tests.unit import test_shell
class TestIntegShellCliV3(test_base.TestInteg):

    def setUp(self):
        super(TestIntegShellCliV3, self).setUp()
        env = {'OS_AUTH_URL': test_base.V3_AUTH_URL, 'OS_PROJECT_DOMAIN_ID': test_shell.DEFAULT_PROJECT_DOMAIN_ID, 'OS_USER_DOMAIN_ID': test_shell.DEFAULT_USER_DOMAIN_ID, 'OS_USERNAME': test_shell.DEFAULT_USERNAME, 'OS_PASSWORD': test_shell.DEFAULT_PASSWORD, 'OS_IDENTITY_API_VERSION': '3'}
        self.useFixture(osc_lib_utils.EnvFixture(copy.deepcopy(env)))
        self.token = test_base.make_v3_token(self.requests_mock)

    def test_shell_args_no_options(self):
        _shell = shell.OpenStackShell()
        _shell.run('extension list'.split())
        self.assertNotEqual(len(self.requests_mock.request_history), 0)
        self.assertEqual(test_base.V3_AUTH_URL, self.requests_mock.request_history[0].url)
        auth_req = self.requests_mock.request_history[1].json()
        self.assertEqual(test_shell.DEFAULT_PROJECT_DOMAIN_ID, auth_req['auth']['identity']['password']['user']['domain']['id'])
        self.assertEqual(test_shell.DEFAULT_USERNAME, auth_req['auth']['identity']['password']['user']['name'])
        self.assertEqual(test_shell.DEFAULT_PASSWORD, auth_req['auth']['identity']['password']['user']['password'])

    def test_shell_args_verify(self):
        _shell = shell.OpenStackShell()
        _shell.run('--verify extension list'.split())
        self.assertNotEqual(len(self.requests_mock.request_history), 0)
        self.assertTrue(self.requests_mock.request_history[0].verify)

    def test_shell_args_insecure(self):
        _shell = shell.OpenStackShell()
        _shell.run('--insecure extension list'.split())
        self.assertNotEqual(len(self.requests_mock.request_history), 0)
        self.assertFalse(self.requests_mock.request_history[0].verify)

    def test_shell_args_cacert(self):
        _shell = shell.OpenStackShell()
        _shell.run('--os-cacert xyzpdq extension list'.split())
        self.assertNotEqual(len(self.requests_mock.request_history), 0)
        self.assertEqual('xyzpdq', self.requests_mock.request_history[0].verify)

    def test_shell_args_cacert_insecure(self):
        _shell = shell.OpenStackShell()
        _shell.run('--os-cacert xyzpdq --insecure extension list'.split())
        self.assertNotEqual(len(self.requests_mock.request_history), 0)
        self.assertFalse(self.requests_mock.request_history[0].verify)