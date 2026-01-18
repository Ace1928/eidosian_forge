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
def test_valid_host_port_string_fail(self):
    invalid_pairs = ['', '10.11.12.13', '172.17.17.1:99999', '290.12.52.80:5673', 'absurd inputs happen', '☁', '☃:8080', 'fe80::1', '[fe80::2]', '<fe80::3>:5673', '[fe80::a:b:c:d]9990', 'fe80:a:b:c:d:e:f:1:2:3:4', 'fe80:a:b:c:d:e:f:g', 'fe80::1:8080', '[fe80:a:b:c:d:e:f:g]:9090', '[a:b:s:u:r:d]:fe80']
    for pair in invalid_pairs:
        self.assertRaises(ValueError, utils.parse_valid_host_port, pair)