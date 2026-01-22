from unittest import mock
import webob
from glance.api.v2 import cached_images
import glance.gateway
from glance import image_cache
from glance import notifier
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
class TestCachedImages(test_utils.BaseTestCase):

    def setUp(self):
        super(TestCachedImages, self).setUp()
        test_controller = FakeController()
        self.controller = test_controller

    def test_get_cached_images(self):
        self.config(image_cache_dir='fake_cache_directory')
        req = webob.Request.blank('')
        req.context = 'test'
        result = self.controller.get_cached_images(req)
        self.assertEqual({'cached_images': [{'image_id': 'test'}]}, result)

    def test_delete_cached_image(self):
        self.config(image_cache_dir='fake_cache_directory')
        req = unit_test_utils.get_fake_request()
        with mock.patch.object(notifier.ImageRepoProxy, 'get') as mock_get:
            mock_get.return_value = FakeImage()
            self.controller.delete_cached_image(req, image_id=UUID4)
            self.assertEqual([UUID4], self.controller.cache.deleted_images)

    def test_delete_cached_images(self):
        self.config(image_cache_dir='fake_cache_directory')
        req = webob.Request.blank('')
        req.context = 'test'
        self.assertEqual({'num_deleted': 1}, self.controller.delete_cached_images(req))
        self.assertEqual(['test'], self.controller.cache.deleted_images)

    def test_get_queued_images(self):
        self.config(image_cache_dir='fake_cache_directory')
        req = webob.Request.blank('')
        req.context = 'test'
        result = self.controller.get_queued_images(req)
        self.assertEqual({'queued_images': {'test': 'passed'}}, result)

    def test_queue_image(self):
        self.config(image_cache_dir='fake_cache_directory')
        req = unit_test_utils.get_fake_request()
        with mock.patch.object(notifier.ImageRepoProxy, 'get') as mock_get:
            mock_get.return_value = FakeImage()
            self.controller.queue_image(req, image_id=UUID4)

    def test_delete_queued_image(self):
        self.config(image_cache_dir='fake_cache_directory')
        req = unit_test_utils.get_fake_request()
        with mock.patch.object(notifier.ImageRepoProxy, 'get') as mock_get:
            mock_get.return_value = FakeImage()
            self.controller.delete_queued_image(req, UUID4)
            self.assertEqual([UUID4], self.controller.cache.deleted_images)

    def test_delete_queued_images(self):
        self.config(image_cache_dir='fake_cache_directory')
        req = webob.Request.blank('')
        req.context = 'test'
        self.assertEqual({'num_deleted': 1}, self.controller.delete_queued_images(req))
        self.assertEqual(['deleted_img'], self.controller.cache.deleted_images)