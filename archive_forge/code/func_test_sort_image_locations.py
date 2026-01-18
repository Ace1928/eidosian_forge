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
def test_sort_image_locations(self):
    enabled_backends = {'rbd1': 'rbd', 'rbd2': 'rbd', 'rbd3': 'rbd'}
    self.config(enabled_backends=enabled_backends)
    store.register_store_opts(CONF)
    self.config(default_backend='rbd1', group='glance_store')
    locations = [{'url': 'rbd://aaaaaaaa/images/id', 'metadata': {'store': 'rbd1'}}, {'url': 'rbd://bbbbbbbb/images/id', 'metadata': {'store': 'rbd2'}}, {'url': 'rbd://cccccccc/images/id', 'metadata': {'store': 'rbd3'}}]
    mp = 'glance.common.utils.glance_store.get_store_from_store_identifier'
    with mock.patch(mp) as mock_get_store:
        mock_store = mock_get_store.return_value
        mock_store.weight = 100
        utils.sort_image_locations(locations)
    self.assertEqual(3, mock_get_store.call_count)