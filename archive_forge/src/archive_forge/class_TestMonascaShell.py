from unittest import mock
from oslotest import base
from monascaclient import shell
class TestMonascaShell(base.BaseTestCase):

    @mock.patch('monascaclient.shell.auth')
    def test_should_use_auth_plugin_option_parser(self, auth):
        auth.build_auth_plugins_option_parser = apop = mock.Mock()
        shell.MonascaShell().run([])
        apop.assert_called_once()

    def test_should_specify_monasca_args(self):
        expected_args = ['--monasca-api-url', '--monasca-api-version', '--monasca_api_url', '--monasca_api_version']
        parser = mock.Mock()
        parser.add_argument = aa = mock.Mock()
        shell.MonascaShell._append_monasca_args(parser)
        aa.assert_called()
        for mc in aa.mock_calls:
            name = mc[1][0]
            self.assertIn(name, expected_args)

    @mock.patch('monascaclient.shell.importutils')
    def test_should_load_commands_based_on_api_version(self, iu):
        iu.import_versioned_module = ivm = mock.Mock()
        instance = shell.MonascaShell()
        instance.options = mock.Mock()
        instance.options.monasca_api_version = version = mock.Mock()
        instance._find_actions = mock.Mock()
        instance._load_commands()
        ivm.assert_called_once_with('monascaclient', version, 'shell')