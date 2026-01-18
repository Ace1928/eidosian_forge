from unittest import mock
from zunclient.common.apiclient import exceptions as apiexec
from zunclient.common import utils as zun_utils
from zunclient.common.websocketclient import exceptions
from zunclient.tests.unit.v1 import shell_test_base
from zunclient.v1 import containers_shell
@mock.patch('zunclient.common.utils.list_containers')
@mock.patch('zunclient.v1.containers.ContainerManager.list')
def test_zun_container_list_success(self, mock_list, mock_list_containers):
    mock_list.return_value = ['container']
    self._test_arg_success('list')
    mock_list_containers.assert_called_once_with(['container'])