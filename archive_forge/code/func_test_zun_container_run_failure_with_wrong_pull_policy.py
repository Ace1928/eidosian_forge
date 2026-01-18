from unittest import mock
from zunclient.common.apiclient import exceptions as apiexec
from zunclient.common import utils as zun_utils
from zunclient.common.websocketclient import exceptions
from zunclient.tests.unit.v1 import shell_test_base
from zunclient.v1 import containers_shell
@mock.patch('zunclient.v1.containers.ContainerManager.run')
def test_zun_container_run_failure_with_wrong_pull_policy(self, mock_run):
    self._test_arg_failure('run --image-pull-policy wrong x', self._invalid_choice_error)
    self.assertFalse(mock_run.called)
    mock_run.assert_not_called()