import contextlib
import hashlib
import io
import math
import os
from unittest import mock
import socket
import sys
import tempfile
import time
import uuid
from keystoneauth1 import exceptions as keystone_exc
from os_brick.initiator import connector
from oslo_concurrency import processutils
from oslo_utils.secretutils import md5
from oslo_utils import units
from glance_store._drivers.cinder import scaleio
from glance_store.common import attachment_state_manager
from glance_store.common import cinder_utils
from glance_store import exceptions
from glance_store import location
from glance_store._drivers.cinder import store as cinder # noqa
from glance_store._drivers.cinder import nfs # noqa
def test_temporary_chown(self):
    fake_stat = mock.MagicMock(st_uid=1)
    with mock.patch.object(os, 'stat', return_value=fake_stat), mock.patch.object(os, 'getuid', return_value=2), mock.patch.object(processutils, 'execute') as mock_execute, mock.patch.object(cinder.Store, 'get_root_helper', return_value='sudo'):
        with self.store.temporary_chown('test'):
            pass
        expected_calls = [mock.call('chown', 2, 'test', run_as_root=True, root_helper='sudo'), mock.call('chown', 1, 'test', run_as_root=True, root_helper='sudo')]
        self.assertEqual(expected_calls, mock_execute.call_args_list)