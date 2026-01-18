import http.client as http
from unittest.mock import patch
from oslo_log.fixture import logging_error as log_fixture
from oslo_policy import policy
from oslo_utils.fixture import uuidsentinel as uuids
import testtools
import webob
import glance.api.middleware.cache
import glance.api.policy
from glance.common import exception
from glance import context
from glance.tests.unit import base
from glance.tests.unit import fixtures as glance_fixtures
from glance.tests.unit import test_policy
from glance.tests.unit import utils as unit_test_utils
def test_stash_cache_request_info(self):
    self.middleware._stash_request_info(self.request, 'asdf', 'GET', 'v2')
    self.assertEqual('asdf', self.request.environ['api.cache.image_id'])
    self.assertEqual('GET', self.request.environ['api.cache.method'])
    self.assertEqual('v2', self.request.environ['api.cache.version'])