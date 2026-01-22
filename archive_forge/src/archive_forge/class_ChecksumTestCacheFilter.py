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
class ChecksumTestCacheFilter(glance.api.middleware.cache.CacheFilter):

    def __init__(self):

        class DummyCache(object):

            def get_caching_iter(self, image_id, image_checksum, app_iter):
                self.image_checksum = image_checksum
        self.cache = DummyCache()
        self.policy = unit_test_utils.FakePolicyEnforcer()