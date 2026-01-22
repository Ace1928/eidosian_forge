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
def _test_get_cinderclient_with_ca_certificates(self, group='glance_store'):
    fake_cert_path = 'fake_cert_path'
    with mock.patch.object(cinder.ksa_session, 'Session') as fake_session, mock.patch.object(cinder.ksa_identity, 'V3Password') as fake_auth_method:
        fake_auth = fake_auth_method()
        self._get_cinderclient_with_user_overriden(group=group, **{'cinder_ca_certificates_file': fake_cert_path})
        fake_session.assert_called_once_with(auth=fake_auth, verify=fake_cert_path)