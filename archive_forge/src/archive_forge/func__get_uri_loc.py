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
def _get_uri_loc(self, fake_volume_uuid, is_multi_store=False):
    if is_multi_store:
        uri = 'cinder://cinder1/%s' % fake_volume_uuid
        loc = location.get_location_from_uri_and_backend(uri, 'cinder1', conf=self.conf)
    else:
        uri = 'cinder://%s' % fake_volume_uuid
        loc = location.get_location_from_uri(uri, conf=self.conf)
    return loc