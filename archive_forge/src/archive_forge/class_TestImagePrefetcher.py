from contextlib import contextmanager
import datetime
import errno
import io
import os
import tempfile
import time
from unittest import mock
import fixtures
import glance_store as store
from oslo_config import cfg
from oslo_utils import fileutils
from oslo_utils import secretutils
from oslo_utils import units
from glance import async_
from glance.common import exception
from glance import context
from glance import gateway as glance_gateway
from glance import image_cache
from glance.image_cache import prefetcher
from glance.tests.unit import utils as unit_test_utils
from glance.tests import utils as test_utils
from glance.tests.utils import skip_if_disabled
from glance.tests.utils import xattr_writes_supported
class TestImagePrefetcher(test_utils.BaseTestCase):

    def setUp(self):
        super(TestImagePrefetcher, self).setUp()
        self.cache_dir = self.useFixture(fixtures.TempDir()).path
        self.config(image_cache_dir=self.cache_dir, image_cache_driver='xattr', image_cache_max_size=5 * units.Ki)
        self.prefetcher = prefetcher.Prefetcher()

    def test_fetch_image_into_cache_without_auth(self):
        with mock.patch.object(self.prefetcher.gateway, 'get_repo') as mock_get:
            self.prefetcher.fetch_image_into_cache('fake-image-id')
            mock_get.assert_called_once_with(mock.ANY)