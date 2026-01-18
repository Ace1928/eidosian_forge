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
def test_process_request_without_download_image_policy(self):
    """
        Test for cache middleware skip processing when request
        context has not 'download_image' role.
        """

    def fake_get_v2_image_metadata(*args, **kwargs):
        image = ImageStub(image_id, request.context.project_id)
        return (image, {'status': 'active', 'properties': {}})
    image_id = 'test1'
    request = webob.Request.blank('/v2/images/%s/file' % image_id)
    request.context = context.RequestContext()
    cache_filter = ProcessRequestTestCacheFilter()
    cache_filter._get_v2_image_metadata = fake_get_v2_image_metadata
    enforcer = self._enforcer_from_rules({'get_image': '', 'download_image': '!'})
    cache_filter.policy = enforcer
    self.assertRaises(webob.exc.HTTPForbidden, cache_filter.process_request, request)