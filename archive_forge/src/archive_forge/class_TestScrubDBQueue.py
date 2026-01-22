from unittest import mock
from unittest.mock import patch
import uuid
import glance_store
from oslo_config import cfg
from glance.common import exception
from glance.db.sqlalchemy import api as db_api
from glance import scrubber
from glance.tests import utils as test_utils
class TestScrubDBQueue(test_utils.BaseTestCase):

    def setUp(self):
        super(TestScrubDBQueue, self).setUp()

    def _create_image_list(self, count):
        images = []
        for x in range(count):
            images.append({'id': x})
        return images

    def test_get_all_images(self):
        scrub_queue = scrubber.ScrubDBQueue()
        images = self._create_image_list(15)
        image_pager = ImagePager(images)

        def make_get_images_detailed(pager):

            def mock_get_images_detailed(ctx, filters, marker=None, limit=None):
                return pager()
            return mock_get_images_detailed
        with patch.object(db_api, 'image_get_all') as _mock_get_images_detailed:
            _mock_get_images_detailed.side_effect = make_get_images_detailed(image_pager)
            actual = list(scrub_queue._get_all_images())
        self.assertEqual(images, actual)

    def test_get_all_images_paged(self):
        scrub_queue = scrubber.ScrubDBQueue()
        images = self._create_image_list(15)
        image_pager = ImagePager(images, page_size=4)

        def make_get_images_detailed(pager):

            def mock_get_images_detailed(ctx, filters, marker=None, limit=None):
                return pager()
            return mock_get_images_detailed
        with patch.object(db_api, 'image_get_all') as _mock_get_images_detailed:
            _mock_get_images_detailed.side_effect = make_get_images_detailed(image_pager)
            actual = list(scrub_queue._get_all_images())
        self.assertEqual(images, actual)