import datetime
import errno
import io
import os
import time
from unittest import mock
from oslo_utils import fileutils
from glance.image_cache.drivers import centralized_db
from glance.tests import functional
def test_open_for_read_with_exception(self):
    self.start_server(enable_cache=True)
    self.driver = centralized_db.Driver()
    self.driver.configure()
    images = self.load_data()
    self.assertFalse(self.driver.is_cached(images['public']))
    path = '/v2/cache/%s' % images['public']
    self.api_put(path)
    self.wait_for_caching(images['public'])
    self.assertTrue(self.driver.is_cached(images['public']))
    self.assertEqual(0, self.driver.get_hit_count(images['public']))
    buff = io.BytesIO()
    try:
        with self.driver.open_for_read(images['public']):
            raise IOError
    except Exception as e:
        self.assertIsInstance(e, IOError)
    self.assertEqual(b'', buff.getvalue())
    self.assertEqual(1, self.driver.get_hit_count(images['public']))