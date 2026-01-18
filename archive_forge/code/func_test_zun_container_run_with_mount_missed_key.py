from unittest import mock
from zunclient.common.apiclient import exceptions as apiexec
from zunclient.common import utils as zun_utils
from zunclient.common.websocketclient import exceptions
from zunclient.tests.unit.v1 import shell_test_base
from zunclient.v1 import containers_shell
def test_zun_container_run_with_mount_missed_key(self):
    self.assertRaisesRegex(apiexec.CommandError, 'Invalid mounts argument', self.shell, 'run --mount source=s x')