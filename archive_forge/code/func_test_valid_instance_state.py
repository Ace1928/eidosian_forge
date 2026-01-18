from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient import extension
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import share_instances
@ddt.data(('2.6', '1234', 'migrating_to'), ('2.6', '1234', 'error'), ('2.6', '1234', 'available'), ('2.7', '1234', 'error_deleting'), ('2.7', '1234', 'deleting'), ('2.7', '1234', 'migrating'))
@ddt.unpack
def test_valid_instance_state(self, microversion, instance, state):
    manager = self._get_manager(microversion)
    if api_versions.APIVersion(microversion) > api_versions.APIVersion('2.6'):
        action_name = 'reset_status'
    else:
        action_name = 'os-reset_status'
    with mock.patch.object(manager, '_action', mock.Mock()):
        manager.reset_state(instance, state)
        manager._action.assert_called_once_with(action_name, instance, {'status': state})