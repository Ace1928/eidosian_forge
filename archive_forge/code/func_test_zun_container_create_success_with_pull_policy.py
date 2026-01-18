from unittest import mock
from zunclient.common.apiclient import exceptions as apiexec
from zunclient.common import utils as zun_utils
from zunclient.common.websocketclient import exceptions
from zunclient.tests.unit.v1 import shell_test_base
from zunclient.v1 import containers_shell
@mock.patch('zunclient.v1.containers_shell._show_container')
@mock.patch('zunclient.v1.containers.ContainerManager.create')
def test_zun_container_create_success_with_pull_policy(self, mock_create, mock_show_container):
    mock_create.return_value = 'container-never'
    self._test_arg_success('create --image-pull-policy never x')
    mock_show_container.assert_called_with('container-never')
    mock_create.assert_called_with(**_get_container_args(image='x', image_pull_policy='never'))
    mock_create.return_value = 'container-always'
    self._test_arg_success('create --image-pull-policy always x')
    mock_show_container.assert_called_with('container-always')
    mock_create.assert_called_with(**_get_container_args(image='x', image_pull_policy='always'))
    mock_create.return_value = 'container-ifnotpresent'
    self._test_arg_success('create --image-pull-policy ifnotpresent x')
    mock_show_container.assert_called_with('container-ifnotpresent')
    mock_create.assert_called_with(**_get_container_args(image='x', image_pull_policy='ifnotpresent'))