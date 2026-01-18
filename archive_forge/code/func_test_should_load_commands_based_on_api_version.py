from unittest import mock
from oslotest import base
from monascaclient import shell
@mock.patch('monascaclient.shell.importutils')
def test_should_load_commands_based_on_api_version(self, iu):
    iu.import_versioned_module = ivm = mock.Mock()
    instance = shell.MonascaShell()
    instance.options = mock.Mock()
    instance.options.monasca_api_version = version = mock.Mock()
    instance._find_actions = mock.Mock()
    instance._load_commands()
    ivm.assert_called_once_with('monascaclient', version, 'shell')