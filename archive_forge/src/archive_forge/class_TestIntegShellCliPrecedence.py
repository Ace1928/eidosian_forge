import copy
from unittest import mock
import fixtures
from osc_lib.tests import utils as osc_lib_utils
from openstackclient import shell
from openstackclient.tests.unit.integ import base as test_base
from openstackclient.tests.unit import test_shell
class TestIntegShellCliPrecedence(test_base.TestInteg):
    """Validate option precedence rules without clouds.yaml

    Global option values may be set in three places:
    * command line options
    * environment variables
    * clouds.yaml

    Verify that the above order is the precedence used,
    i.e. a command line option overrides all others, etc
    """

    def setUp(self):
        super(TestIntegShellCliPrecedence, self).setUp()
        env = {'OS_AUTH_URL': test_base.V3_AUTH_URL, 'OS_PROJECT_DOMAIN_ID': test_shell.DEFAULT_PROJECT_DOMAIN_ID, 'OS_USER_DOMAIN_ID': test_shell.DEFAULT_USER_DOMAIN_ID, 'OS_USERNAME': test_shell.DEFAULT_USERNAME, 'OS_IDENTITY_API_VERSION': '3'}
        self.useFixture(osc_lib_utils.EnvFixture(copy.deepcopy(env)))
        self.token = test_base.make_v3_token(self.requests_mock)
        test_shell.PUBLIC_1['public-clouds']['megadodo']['auth']['auth_url'] = test_base.V3_AUTH_URL

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