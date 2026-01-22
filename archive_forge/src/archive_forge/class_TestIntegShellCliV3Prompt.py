import copy
from unittest import mock
import fixtures
from osc_lib.tests import utils as osc_lib_utils
from openstackclient import shell
from openstackclient.tests.unit.integ import base as test_base
from openstackclient.tests.unit import test_shell
class TestIntegShellCliV3Prompt(test_base.TestInteg):

    def setUp(self):
        super(TestIntegShellCliV3Prompt, self).setUp()
        env = {'OS_AUTH_URL': test_base.V3_AUTH_URL, 'OS_PROJECT_DOMAIN_ID': test_shell.DEFAULT_PROJECT_DOMAIN_ID, 'OS_USER_DOMAIN_ID': test_shell.DEFAULT_USER_DOMAIN_ID, 'OS_USERNAME': test_shell.DEFAULT_USERNAME, 'OS_IDENTITY_API_VERSION': '3'}
        self.useFixture(osc_lib_utils.EnvFixture(copy.deepcopy(env)))
        self.token = test_base.make_v3_token(self.requests_mock)

    @mock.patch('osc_lib.shell.prompt_for_password')
    def test_shell_callback(self, mock_prompt):
        mock_prompt.return_value = 'qaz'
        _shell = shell.OpenStackShell()
        _shell.run('extension list'.split())
        self.assertNotEqual(len(self.requests_mock.request_history), 0)
        self.assertEqual(mock_prompt, _shell.cloud._openstack_config._pw_callback)
        auth_req = self.requests_mock.request_history[1].json()
        self.assertEqual('qaz', auth_req['auth']['identity']['password']['user']['password'])