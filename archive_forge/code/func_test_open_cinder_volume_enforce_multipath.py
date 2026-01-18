import errno
import io
from unittest import mock
import sys
import uuid
from oslo_utils import units
from glance_store import exceptions
from glance_store.tests import base
from glance_store.tests.unit.cinder import test_cinder_base
from glance_store.tests.unit import test_store_capabilities
from glance_store._drivers.cinder import store as cinder # noqa
def test_open_cinder_volume_enforce_multipath(self):
    self.config(cinder_use_multipath=True)
    self.config(cinder_enforce_multipath=True)
    self._test_open_cinder_volume('wb', 'rw', None, multipath_supported=True, enforce_multipath=True)