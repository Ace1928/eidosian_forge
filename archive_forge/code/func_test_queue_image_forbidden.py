from unittest import mock
import webob
from glance.api.v2 import cached_images
import glance.gateway
from glance import image_cache
from glance import notifier
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_queue_image_forbidden(self):
    self.config(image_cache_dir='fake_cache_directory')
    self.controller.policy.rules = {'manage_image_cache': False}
    req = unit_test_utils.get_fake_request()
    with mock.patch.object(notifier.ImageRepoProxy, 'get') as mock_get:
        mock_get.return_value = FakeImage()
        self.assertRaises(webob.exc.HTTPForbidden, self.controller.queue_image, req, image_id=UUID4)