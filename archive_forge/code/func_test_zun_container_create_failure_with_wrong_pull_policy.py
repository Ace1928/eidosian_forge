from unittest import mock
from zunclient.common.apiclient import exceptions as apiexec
from zunclient.common import utils as zun_utils
from zunclient.common.websocketclient import exceptions
from zunclient.tests.unit.v1 import shell_test_base
from zunclient.v1 import containers_shell
@mock.patch('zunclient.v1.containers.ContainerManager.create')
def test_zun_container_create_failure_with_wrong_pull_policy(self, mock_create):
    self._test_arg_failure('create --image-pull-policy wrong x ', self._invalid_choice_error)
    self.assertFalse(mock_create.called)
    mock_create.assert_not_called()