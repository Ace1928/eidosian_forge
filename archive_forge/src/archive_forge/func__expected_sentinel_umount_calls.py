import sys
from unittest import mock
import fixtures
from oslo_concurrency import processutils
from oslo_config import cfg
from oslotest import base
from glance_store import exceptions
@staticmethod
def _expected_sentinel_umount_calls(mountpoint=mock.sentinel.mountpoint):
    return [mock.call('umount', mountpoint, attempts=3, delay_on_retry=True, root_helper=mock.sentinel.rootwrap_helper, run_as_root=True)]