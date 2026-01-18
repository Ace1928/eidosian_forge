import sys
from unittest import mock
import urllib.error
from glance_store import exceptions as store_exceptions
from oslo_config import cfg
from oslo_utils import units
import taskflow
import glance.async_.flows.api_image_import as import_flow
from glance.common import exception
from glance.common.scripts.image_import import main as image_import
from glance import context
from glance.domain import ExtraProperties
from glance import gateway
import glance.tests.utils as test_utils
from cursive import exception as cursive_exception
@mock.patch.object(image_import, 'set_image_data')
def test_set_image_data_with_callback(self, mock_sid):

    def fake_set_image_data(image, uri, task_id, backend=None, set_active=False, callback=None):
        callback(mock.sentinel.chunk, mock.sentinel.total)
    mock_sid.side_effect = fake_set_image_data
    callback = mock.MagicMock()
    self.actions.set_image_data(mock.sentinel.uri, mock.sentinel.task_id, mock.sentinel.backend, mock.sentinel.set_active, callback=callback)
    callback.assert_called_once_with(self.actions, mock.sentinel.chunk, mock.sentinel.total)