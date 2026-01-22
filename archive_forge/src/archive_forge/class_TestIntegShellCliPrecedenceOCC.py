import copy
from unittest import mock
import fixtures
from osc_lib.tests import utils as osc_lib_utils
from openstackclient import shell
from openstackclient.tests.unit.integ import base as test_base
from openstackclient.tests.unit import test_shell
class TestIntegShellCliPrecedenceOCC(test_base.TestInteg):
    """Validate option precedence rules with clouds.yaml

    Global option values may be set in three places:
    * command line options
    * environment variables
    * clouds.yaml

    Verify that the above order is the precedence used,
    i.e. a command line option overrides all others, etc
    """

    def setUp(self):
        super(TestIntegShellCliPrecedenceOCC, self).setUp()
        env = {'OS_CLOUD': 'megacloud', 'OS_AUTH_URL': test_base.V3_AUTH_URL, 'OS_PROJECT_DOMAIN_ID': test_shell.DEFAULT_PROJECT_DOMAIN_ID, 'OS_USER_DOMAIN_ID': test_shell.DEFAULT_USER_DOMAIN_ID, 'OS_USERNAME': test_shell.DEFAULT_USERNAME, 'OS_IDENTITY_API_VERSION': '3', 'OS_CLOUD_NAME': 'qaz'}
        self.useFixture(osc_lib_utils.EnvFixture(copy.deepcopy(env)))
        self.token = test_base.make_v3_token(self.requests_mock)
        test_shell.PUBLIC_1['public-clouds']['megadodo']['auth']['auth_url'] = test_base.V3_AUTH_URL

    def get_temp_file_path(self, filename):
        """Returns an absolute path for a temporary file.

        :param filename: filename
        :type filename: string
        :returns: absolute file path string
        """
        temp_dir = self.useFixture(fixtures.TempDir())
        return temp_dir.join(filename)

    @mock.patch('openstack.config.loader.OpenStackConfig._load_vendor_file')
    @mock.patch('openstack.config.loader.OpenStackConfig._load_config_file')
    def test_shell_args_precedence_1(self, config_mock, vendor_mock):
        """Precedence run 1

        Run 1 has --os-password on CLI
        """

        def config_mock_return():
            log_file = self.get_temp_file_path('test_log_file')
            cloud2 = test_shell.get_cloud(log_file)
            return ('file.yaml', cloud2)
        config_mock.side_effect = config_mock_return

        def vendor_mock_return():
            return ('file.yaml', copy.deepcopy(test_shell.PUBLIC_1))
        vendor_mock.side_effect = vendor_mock_return
        _shell = shell.OpenStackShell()
        _shell.run('--os-password qaz extension list'.split())
        self.assertNotEqual(len(self.requests_mock.request_history), 0)
        self.assertEqual(test_base.V3_AUTH_URL, self.requests_mock.request_history[0].url)
        auth_req = self.requests_mock.request_history[1].json()
        self.assertEqual('heart-o-gold', auth_req['auth']['scope']['project']['name'])
        self.assertEqual('qaz', auth_req['auth']['identity']['password']['user']['password'])
        self.assertEqual(test_shell.DEFAULT_USER_DOMAIN_ID, auth_req['auth']['identity']['password']['user']['domain']['id'])
        self.assertEqual(test_shell.DEFAULT_USERNAME, auth_req['auth']['identity']['password']['user']['name'])

    @mock.patch('openstack.config.loader.OpenStackConfig._load_vendor_file')
    @mock.patch('openstack.config.loader.OpenStackConfig._load_config_file')
    def test_shell_args_precedence_2(self, config_mock, vendor_mock):
        """Precedence run 2

        Run 2 has --os-username, --os-password, --os-project-domain-id on CLI
        """

        def config_mock_return():
            log_file = self.get_temp_file_path('test_log_file')
            cloud2 = test_shell.get_cloud(log_file)
            return ('file.yaml', cloud2)
        config_mock.side_effect = config_mock_return

        def vendor_mock_return():
            return ('file.yaml', copy.deepcopy(test_shell.PUBLIC_1))
        vendor_mock.side_effect = vendor_mock_return
        _shell = shell.OpenStackShell()
        _shell.run('--os-username zarquon --os-password qaz --os-project-domain-id 5678 extension list'.split())
        self.assertNotEqual(len(self.requests_mock.request_history), 0)
        self.assertEqual(test_base.V3_AUTH_URL, self.requests_mock.request_history[0].url)
        auth_req = self.requests_mock.request_history[1].json()
        self.assertEqual('5678', auth_req['auth']['scope']['project']['domain']['id'])
        self.assertEqual('zarquon', auth_req['auth']['identity']['password']['user']['name'])