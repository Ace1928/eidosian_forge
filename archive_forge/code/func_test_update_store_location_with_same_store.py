import io
import tempfile
from unittest import mock
import glance_store as store
from glance_store._drivers import cinder
from oslo_config import cfg
from oslo_log import log as logging
import webob
from glance.common import exception
from glance.common import store_utils
from glance.common import utils
from glance.tests.unit import base
from glance.tests import utils as test_utils
def test_update_store_location_with_same_store(self):
    enabled_backends = {'rbd1': 'rbd', 'rbd2': 'rbd'}
    self.config(enabled_backends=enabled_backends)
    self._test_update_store_in_location({'store': 'rbd1'}, 'rbd1', 'rbd1', store_id_call_count=0, save_call_count=0)