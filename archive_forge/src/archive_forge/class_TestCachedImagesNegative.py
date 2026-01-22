from unittest import mock
import webob
from glance.api.v2 import cached_images
import glance.gateway
from glance import image_cache
from glance import notifier
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
class TestCachedImagesNegative(test_utils.BaseTestCase):

    def setUp(self):
        super(TestCachedImagesNegative, self).setUp()
        test_controller = FakeController()
        self.controller = test_controller

    def test_get_cached_images_disabled(self):
        req = webob.Request.blank('')
        req.context = 'test'
        self.assertRaises(webob.exc.HTTPNotFound, self.controller.get_cached_images, req)

    def test_get_cached_images_forbidden(self):
        self.config(image_cache_dir='fake_cache_directory')
        self.controller.policy.rules = {'manage_image_cache': False}
        req = unit_test_utils.get_fake_request()
        with mock.patch.object(notifier.ImageRepoProxy, 'get') as mock_get:
            mock_get.return_value = FakeImage()
            self.assertRaises(webob.exc.HTTPForbidden, self.controller.get_cached_images, req)

    def test_delete_cached_image_disabled(self):
        req = webob.Request.blank('')
        req.context = 'test'
        self.assertRaises(webob.exc.HTTPNotFound, self.controller.delete_cached_image, req, image_id='test')

    def test_delete_cached_image_forbidden(self):
        self.config(image_cache_dir='fake_cache_directory')
        self.controller.policy.rules = {'manage_image_cache': False}
        req = unit_test_utils.get_fake_request()
        with mock.patch.object(notifier.ImageRepoProxy, 'get') as mock_get:
            mock_get.return_value = FakeImage()
            self.assertRaises(webob.exc.HTTPForbidden, self.controller.delete_cached_image, req, image_id=UUID4)

    def test_delete_cached_images_disabled(self):
        req = webob.Request.blank('')
        req.context = 'test'
        self.assertRaises(webob.exc.HTTPNotFound, self.controller.delete_cached_images, req)

    def test_delete_cached_images_forbidden(self):
        self.config(image_cache_dir='fake_cache_directory')
        self.controller.policy.rules = {'manage_image_cache': False}
        req = unit_test_utils.get_fake_request()
        with mock.patch.object(notifier.ImageRepoProxy, 'get') as mock_get:
            mock_get.return_value = FakeImage()
            self.assertRaises(webob.exc.HTTPForbidden, self.controller.delete_cached_images, req)

    def test_get_queued_images_disabled(self):
        req = webob.Request.blank('')
        req.context = 'test'
        self.assertRaises(webob.exc.HTTPNotFound, self.controller.get_queued_images, req)

    def test_get_queued_images_forbidden(self):
        self.config(image_cache_dir='fake_cache_directory')
        self.controller.policy.rules = {'manage_image_cache': False}
        req = unit_test_utils.get_fake_request()
        with mock.patch.object(notifier.ImageRepoProxy, 'get') as mock_get:
            mock_get.return_value = FakeImage()
            self.assertRaises(webob.exc.HTTPForbidden, self.controller.get_queued_images, req)

    def test_queue_image_disabled(self):
        req = webob.Request.blank('')
        req.context = 'test'
        self.assertRaises(webob.exc.HTTPNotFound, self.controller.queue_image, req, image_id='test1')

    def test_queue_image_forbidden(self):
        self.config(image_cache_dir='fake_cache_directory')
        self.controller.policy.rules = {'manage_image_cache': False}
        req = unit_test_utils.get_fake_request()
        with mock.patch.object(notifier.ImageRepoProxy, 'get') as mock_get:
            mock_get.return_value = FakeImage()
            self.assertRaises(webob.exc.HTTPForbidden, self.controller.queue_image, req, image_id=UUID4)

    def test_delete_queued_image_disabled(self):
        req = webob.Request.blank('')
        req.context = 'test'
        self.assertRaises(webob.exc.HTTPNotFound, self.controller.delete_queued_image, req, image_id='test1')

    def test_delete_queued_image_forbidden(self):
        self.config(image_cache_dir='fake_cache_directory')
        self.controller.policy.rules = {'manage_image_cache': False}
        req = unit_test_utils.get_fake_request()
        with mock.patch.object(notifier.ImageRepoProxy, 'get') as mock_get:
            mock_get.return_value = FakeImage()
            self.assertRaises(webob.exc.HTTPForbidden, self.controller.delete_queued_image, req, image_id=UUID4)

    def test_delete_queued_images_disabled(self):
        req = webob.Request.blank('')
        req.context = 'test'
        self.assertRaises(webob.exc.HTTPNotFound, self.controller.delete_queued_images, req)

    def test_delete_queued_images_forbidden(self):
        self.config(image_cache_dir='fake_cache_directory')
        self.controller.policy.rules = {'manage_image_cache': False}
        req = unit_test_utils.get_fake_request()
        with mock.patch.object(notifier.ImageRepoProxy, 'get') as mock_get:
            mock_get.return_value = FakeImage()
            self.assertRaises(webob.exc.HTTPForbidden, self.controller.delete_queued_images, req)

    def test_delete_cache_entry_forbidden(self):
        self.config(image_cache_dir='fake_cache_directory')
        self.controller.policy.rules = {'cache_delete': False}
        req = unit_test_utils.get_fake_request()
        with mock.patch.object(notifier.ImageRepoProxy, 'get') as mock_get:
            mock_get.return_value = FakeImage()
            self.assertRaises(webob.exc.HTTPForbidden, self.controller.delete_cache_entry, req, image_id=UUID4)

    def test_delete_cache_entry_disabled(self):
        req = unit_test_utils.get_fake_request()
        with mock.patch.object(notifier.ImageRepoProxy, 'get') as mock_get:
            mock_get.return_value = FakeImage()
        self.assertRaises(webob.exc.HTTPNotFound, self.controller.delete_cache_entry, req, image_id=UUID4)

    def test_delete_non_existing_cache_entries(self):
        self.config(image_cache_dir='fake_cache_directory')
        req = unit_test_utils.get_fake_request()
        self.assertRaises(webob.exc.HTTPNotFound, self.controller.delete_cache_entry, req, image_id='non-existing-queued-image')

    def test_clear_cache_forbidden(self):
        self.config(image_cache_dir='fake_cache_directory')
        self.controller.policy.rules = {'cache_delete': False}
        req = unit_test_utils.get_fake_request()
        self.assertRaises(webob.exc.HTTPForbidden, self.controller.clear_cache, req)

    def test_clear_cache_disabled(self):
        req = webob.Request.blank('')
        req.context = 'test'
        self.assertRaises(webob.exc.HTTPNotFound, self.controller.clear_cache, req)

    def test_cache_clear_invalid_target(self):
        self.config(image_cache_dir='fake_cache_directory')
        req = unit_test_utils.get_fake_request()
        req.headers.update({'x-image-cache-clear-target': 'invalid'})
        self.assertRaises(webob.exc.HTTPBadRequest, self.controller.clear_cache, req)

    def test_get_cache_state_disabled(self):
        req = webob.Request.blank('')
        req.context = 'test'
        self.assertRaises(webob.exc.HTTPNotFound, self.controller.get_cache_state, req)

    def test_get_cache_state_forbidden(self):
        self.config(image_cache_dir='fake_cache_directory')
        self.controller.policy.rules = {'cache_list': False}
        req = unit_test_utils.get_fake_request()
        with mock.patch.object(notifier.ImageRepoProxy, 'get') as mock_get:
            mock_get.return_value = FakeImage()
            self.assertRaises(webob.exc.HTTPForbidden, self.controller.get_cache_state, req)

    def test_queue_image_from_api_disabled(self):
        req = webob.Request.blank('')
        req.context = 'test'
        self.assertRaises(webob.exc.HTTPNotFound, self.controller.queue_image_from_api, req, image_id='test1')

    def test_queue_image_from_api_forbidden(self):
        self.config(image_cache_dir='fake_cache_directory')
        self.controller.policy.rules = {'cache_image': False}
        req = unit_test_utils.get_fake_request()
        with mock.patch.object(notifier.ImageRepoProxy, 'get') as mock_get:
            mock_get.return_value = FakeImage()
            self.assertRaises(webob.exc.HTTPForbidden, self.controller.queue_image_from_api, req, image_id=UUID4)

    def test_non_active_image_for_queue_api(self):
        self.config(image_cache_dir='fake_cache_directory')
        req = unit_test_utils.get_fake_request()
        for status in ('saving', 'queued', 'pending_delete', 'deactivated', 'importing', 'uploading'):
            with mock.patch.object(notifier.ImageRepoProxy, 'get') as mock_get:
                mock_get.return_value = FakeImage(status=status)
                self.assertRaises(webob.exc.HTTPBadRequest, self.controller.queue_image_from_api, req, image_id=UUID4)

    def test_queue_api_non_existing_image_(self):
        self.config(image_cache_dir='fake_cache_directory')
        req = unit_test_utils.get_fake_request()
        self.assertRaises(webob.exc.HTTPNotFound, self.controller.queue_image_from_api, req, image_id='non-existing-image-id')