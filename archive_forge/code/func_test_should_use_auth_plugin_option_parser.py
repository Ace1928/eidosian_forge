from unittest import mock
from oslotest import base
from monascaclient import shell
@mock.patch('monascaclient.shell.auth')
def test_should_use_auth_plugin_option_parser(self, auth):
    auth.build_auth_plugins_option_parser = apop = mock.Mock()
    shell.MonascaShell().run([])
    apop.assert_called_once()