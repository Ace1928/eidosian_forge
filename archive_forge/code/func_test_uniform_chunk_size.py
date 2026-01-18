from unittest import mock
from oslo_log.fixture import logging_error as log_fixture
import testtools
import webob
import glance.api.common
from glance.common import exception
from glance.tests.unit import fixtures as glance_fixtures
def test_uniform_chunk_size(self):
    resp = self._get_webob_response()
    meta = self._get_image_metadata()
    checked_image = glance.api.common.size_checked_iter(resp, meta, 4, ['AB', 'CD'], None)
    self.assertEqual('AB', next(checked_image))
    self.assertEqual('CD', next(checked_image))
    self.assertRaises(StopIteration, next, checked_image)