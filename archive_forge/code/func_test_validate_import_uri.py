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
def test_validate_import_uri(self):
    self.assertTrue(utils.validate_import_uri('http://foo.com'))
    self.config(allowed_schemes=['http'], group='import_filtering_opts')
    self.config(allowed_hosts=['example.com'], group='import_filtering_opts')
    self.assertTrue(utils.validate_import_uri('http://example.com'))
    self.config(allowed_ports=['8080'], group='import_filtering_opts')
    self.assertTrue(utils.validate_import_uri('http://example.com:8080'))