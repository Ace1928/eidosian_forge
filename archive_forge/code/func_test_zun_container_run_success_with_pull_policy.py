from unittest import mock
from zunclient.common.apiclient import exceptions as apiexec
from zunclient.common import utils as zun_utils
from zunclient.common.websocketclient import exceptions
from zunclient.tests.unit.v1 import shell_test_base
from zunclient.v1 import containers_shell
@mock.patch('zunclient.v1.containers_shell._show_container')
@mock.patch('zunclient.v1.containers.ContainerManager.run')
def test_zun_container_run_success_with_pull_policy(self, mock_run, mock_show_container):
    mock_run.return_value = 'container-never'
    self._test_arg_success('run --image-pull-policy never x')
    mock_show_container.assert_called_with('container-never')
    mock_run.assert_called_with(**_get_container_args(image='x', image_pull_policy='never'))
    mock_run.return_value = 'container-always'
    self._test_arg_success('run --image-pull-policy always x ')
    mock_show_container.assert_called_with('container-always')
    mock_run.assert_called_with(**_get_container_args(image='x', image_pull_policy='always'))
    mock_run.return_value = 'container-ifnotpresent'
    self._test_arg_success('run --image-pull-policy ifnotpresent x')
    mock_show_container.assert_called_with('container-ifnotpresent')
    mock_run.assert_called_with(**_get_container_args(image='x', image_pull_policy='ifnotpresent'))