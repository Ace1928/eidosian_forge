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
def _test_verify_metadata_zero_size(self, image_meta):
    """
        Test verify_metadata updates metadata with cached image size for images
        with 0 size.

        :param image_meta: Image metadata, which may be either an ImageTarget
                           instance or a legacy v1 dict.
        """
    image_size = 1
    cache_filter = ProcessRequestTestCacheFilter()
    with patch.object(cache_filter.cache, 'get_image_size', return_value=image_size):
        cache_filter._verify_metadata(image_meta)
    self.assertEqual(image_size, image_meta['size'])