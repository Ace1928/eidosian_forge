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
def _setup_fixture_file(self):
    FIXTURE_FILE = io.BytesIO(FIXTURE_DATA)
    self.assertFalse(self.cache.is_cached(1))
    self.assertTrue(self.cache.cache_image_file(1, FIXTURE_FILE))
    self.assertTrue(self.cache.is_cached(1))