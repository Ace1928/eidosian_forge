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
@mock.patch.object(time, 'sleep')
def test_wait_volume_status_timeout(self, mock_sleep):
    fake_manager = mock.MagicMock(get=mock.Mock())
    volume_available = mock.MagicMock(manager=fake_manager, id='fake-id', status='available')
    fake_manager.get.return_value = volume_available
    self.assertRaises(exceptions.BackendException, self.store._wait_volume_status, volume_available, 'available', 'in-use')
    fake_manager.get.assert_called_with('fake-id')