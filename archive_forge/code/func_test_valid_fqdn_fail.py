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
def test_valid_fqdn_fail(self):
    invalid_inputs = ['localhost', '192.168.0.1', '999.88.77.6', '☃.local', 'glance02.stack42']
    for input_str in invalid_inputs:
        self.assertFalse(utils.is_valid_fqdn(input_str))