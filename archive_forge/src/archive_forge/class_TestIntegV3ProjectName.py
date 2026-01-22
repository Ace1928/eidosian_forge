import copy
from osc_lib.tests import utils as osc_lib_utils
from openstackclient import shell
from openstackclient.tests.unit.integ import base as test_base
from openstackclient.tests.unit import test_shell
class TestIntegV3ProjectName(test_base.TestInteg):

    def setUp(self):
        super(TestIntegV3ProjectName, self).setUp()
        env = {'OS_AUTH_URL': test_base.V3_AUTH_URL, 'OS_PROJECT_NAME': test_shell.DEFAULT_PROJECT_NAME, 'OS_USERNAME': test_shell.DEFAULT_USERNAME, 'OS_PASSWORD': test_shell.DEFAULT_PASSWORD, 'OS_IDENTITY_API_VERSION': '3'}
        self.useFixture(osc_lib_utils.EnvFixture(copy.deepcopy(env)))
        self.token = test_base.make_v3_token(self.requests_mock)

    def test_project_name_env(self):
        _shell = shell.OpenStackShell()
        _shell.run('extension list'.split())
        self.assertNotEqual(len(self.requests_mock.request_history), 0)
        self.assertEqual(test_base.V3_AUTH_URL, self.requests_mock.request_history[0].url)
        auth_req = self.requests_mock.request_history[1].json()
        self.assertEqual(test_shell.DEFAULT_PROJECT_NAME, auth_req['auth']['scope']['project']['name'])
        self.assertIsNone(auth_req['auth'].get('tenantId', None))
        self.assertIsNone(auth_req['auth'].get('tenantName', None))

    def test_project_name_arg(self):
        _shell = shell.OpenStackShell()
        _shell.run('--os-project-name wsx extension list'.split())
        self.assertNotEqual(len(self.requests_mock.request_history), 0)
        self.assertEqual(test_base.V3_AUTH_URL, self.requests_mock.request_history[0].url)
        auth_req = self.requests_mock.request_history[1].json()
        self.assertEqual('wsx', auth_req['auth']['scope']['project']['name'])
        self.assertIsNone(auth_req['auth'].get('tenantId', None))
        self.assertIsNone(auth_req['auth'].get('tenantName', None))