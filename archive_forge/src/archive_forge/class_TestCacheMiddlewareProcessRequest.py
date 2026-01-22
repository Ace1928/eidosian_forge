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
class TestCacheMiddlewareProcessRequest(base.IsolatedUnitTest):

    def _enforcer_from_rules(self, unparsed_rules):
        rules = policy.Rules.from_dict(unparsed_rules)
        enforcer = glance.api.policy.Enforcer(suppress_deprecation_warnings=True)
        enforcer.set_rules(rules, overwrite=True)
        return enforcer

    def test_verify_metadata_deleted_image(self):
        """
        Test verify_metadata raises exception.NotFound for a deleted image
        """
        image_meta = {'status': 'deleted', 'is_public': True, 'deleted': True}
        cache_filter = ProcessRequestTestCacheFilter()
        self.assertRaises(exception.NotFound, cache_filter._verify_metadata, image_meta)

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

    def test_verify_metadata_zero_size(self):
        """
        Test verify_metadata updates metadata with cached image size for images
        with 0 size
        """
        image_meta = {'size': 0, 'deleted': False, 'id': 'test1', 'status': 'active'}
        self._test_verify_metadata_zero_size(image_meta)

    def test_verify_metadata_is_image_target_instance_with_zero_size(self):
        """
        Test verify_metadata updates metadata which is ImageTarget instance
        """
        image = ImageStub('test1', uuids.owner)
        image.size = 0
        image_meta = glance.api.policy.ImageTarget(image)
        self._test_verify_metadata_zero_size(image_meta)

    def test_v2_process_request_response_headers(self):

        def dummy_img_iterator():
            for i in range(3):
                yield i
        image_id = 'test1'
        request = webob.Request.blank('/v2/images/test1/file')
        request.context = context.RequestContext()
        image_meta = {'id': image_id, 'name': 'fake_image', 'status': 'active', 'created_at': '', 'min_disk': '10G', 'min_ram': '1024M', 'protected': False, 'locations': '', 'checksum': 'c1234', 'owner': '', 'disk_format': 'raw', 'container_format': 'bare', 'size': '123456789', 'virtual_size': '123456789', 'is_public': 'public', 'deleted': False, 'updated_at': '', 'properties': {}}
        image = ImageStub(image_id, request.context.project_id)
        request.environ['api.cache.image'] = image
        for k, v in image_meta.items():
            setattr(image, k, v)
        cache_filter = ProcessRequestTestCacheFilter()
        response = cache_filter._process_v2_request(request, image_id, dummy_img_iterator, image_meta)
        self.assertEqual('application/octet-stream', response.headers['Content-Type'])
        self.assertEqual('c1234', response.headers['Content-MD5'])
        self.assertEqual('123456789', response.headers['Content-Length'])

    def test_v2_process_request_without_checksum(self):

        def dummy_img_iterator():
            for i in range(3):
                yield i
        image_id = 'test1'
        request = webob.Request.blank('/v2/images/test1/file')
        request.context = context.RequestContext()
        image = ImageStub(image_id, request.context.project_id)
        image.checksum = None
        request.environ['api.cache.image'] = image
        image_meta = {'id': image_id, 'name': 'fake_image', 'status': 'active', 'size': '123456789'}
        cache_filter = ProcessRequestTestCacheFilter()
        response = cache_filter._process_v2_request(request, image_id, dummy_img_iterator, image_meta)
        self.assertNotIn('Content-MD5', response.headers.keys())

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

    def test_v2_process_request_download_restricted(self):
        """
        Test process_request for v2 api where _member_ role not able to
        download the image with custom property.
        """
        image_id = 'test1'
        extra_properties = {'x_test_key': 'test_1234'}

        def fake_get_v2_image_metadata(*args, **kwargs):
            image = ImageStub(image_id, request.context.project_id, extra_properties=extra_properties)
            request.environ['api.cache.image'] = image
            return (image, glance.api.policy.ImageTarget(image))
        enforcer = self._enforcer_from_rules({'restricted': "not ('test_1234':%(x_test_key)s and role:_member_)", 'download_image': 'role:admin or rule:restricted', 'get_image': ''})
        request = webob.Request.blank('/v2/images/test1/file')
        request.context = context.RequestContext(roles=['_member_'])
        cache_filter = ProcessRequestTestCacheFilter()
        cache_filter._get_v2_image_metadata = fake_get_v2_image_metadata
        cache_filter.policy = enforcer
        self.assertRaises(webob.exc.HTTPForbidden, cache_filter.process_request, request)

    def test_v2_process_request_download_permitted(self):
        """
        Test process_request for v2 api where member role able to
        download the image with custom property.
        """
        image_id = 'test1'
        extra_properties = {'x_test_key': 'test_1234'}

        def fake_get_v2_image_metadata(*args, **kwargs):
            image = ImageStub(image_id, request.context.project_id, extra_properties=extra_properties)
            request.environ['api.cache.image'] = image
            return (image, glance.api.policy.ImageTarget(image))
        request = webob.Request.blank('/v2/images/test1/file')
        request.context = context.RequestContext(roles=['member'])
        cache_filter = ProcessRequestTestCacheFilter()
        cache_filter._get_v2_image_metadata = fake_get_v2_image_metadata
        rules = {'restricted': "not ('test_1234':%(x_test_key)s and role:_member_)", 'download_image': 'role:admin or rule:restricted'}
        self.set_policy_rules(rules)
        cache_filter.policy = glance.api.policy.Enforcer(suppress_deprecation_warnings=True)
        actual = cache_filter.process_request(request)
        self.assertTrue(actual)