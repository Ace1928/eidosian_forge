import io
import tempfile
from unittest import mock
import glance_store as store
from glance_store._drivers import cinder
from oslo_config import cfg
from oslo_log import log as logging
import webob
from glance.common import exception
from glance.common import store_utils
from glance.common import utils
from glance.tests.unit import base
from glance.tests import utils as test_utils
def _test_reader_chunked(self, chunk_size, read_size, max_iterations=5):
    generator = self._create_generator(chunk_size, max_iterations)
    reader = utils.CooperativeReader(generator)
    result = bytearray()
    while True:
        data = reader.read(read_size)
        if len(data) == 0:
            break
        self.assertLessEqual(len(data), read_size)
        result += data
    expected = b'a' * chunk_size + b'b' * chunk_size + b'c' * chunk_size + b'a' * chunk_size + b'b' * chunk_size
    self.assertEqual(expected, bytes(result))