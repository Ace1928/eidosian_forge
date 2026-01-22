from unittest import mock
import oslo_policy.policy
from glance.api import policy
from glance.image_cache import prefetcher
from glance.tests import functional
def _create_upload_and_cache(self, cache_image=False, expected_code=200):
    image_id = self._create_and_upload()
    path = '/v2/queued_images/%s' % image_id
    response = self.api_put(path)
    self.assertEqual(expected_code, response.status_code)
    if cache_image:
        cache_prefetcher = prefetcher.Prefetcher()
        cache_prefetcher.run()
    return image_id