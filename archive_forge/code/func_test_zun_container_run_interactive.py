from unittest import mock
from zunclient.common.apiclient import exceptions as apiexec
from zunclient.common import utils as zun_utils
from zunclient.common.websocketclient import exceptions
from zunclient.tests.unit.v1 import shell_test_base
from zunclient.v1 import containers_shell
@mock.patch('zunclient.v1.containers.ContainerManager.get')
@mock.patch('zunclient.v1.containers_shell._show_container')
@mock.patch('zunclient.v1.containers.ContainerManager.run')
def test_zun_container_run_interactive(self, mock_run, mock_show_container, mock_get_container):
    fake_container = mock.MagicMock()
    fake_container.uuid = 'fake_uuid'
    mock_run.return_value = fake_container
    fake_container.status = 'Error'
    mock_get_container.return_value = fake_container
    self.assertRaises(exceptions.ContainerStateError, self.shell, 'run -i x ')