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
@mock.patch('glance.async_.flows.api_image_import.LOG.debug')
@mock.patch('oslo_utils.timeutils.now')
def test_status_callback_limits_rate(self, mock_now, mock_log):
    img_repo = mock.MagicMock()
    task_repo = mock.MagicMock()
    task_repo.get.return_value.status = 'processing'
    wrapper = import_flow.ImportActionWrapper(img_repo, IMAGE_ID1, TASK_ID1)
    image_import = import_flow._ImportToStore(TASK_ID1, TASK_TYPE, task_repo, wrapper, 'http://url', None, False, True)
    expected_calls = []
    log_call = mock.call('Image import %(image_id)s copied %(copied)i MiB', {'image_id': IMAGE_ID1, 'copied': 0})
    action = mock.MagicMock(image_id=IMAGE_ID1)
    mock_now.return_value = 1000
    image_import._status_callback(action, 32, 32)
    expected_calls.append(log_call)
    mock_log.assert_has_calls(expected_calls)
    image_import._status_callback(action, 32, 64)
    mock_log.assert_has_calls(expected_calls)
    mock_now.return_value += 190
    image_import._status_callback(action, 32, 96)
    mock_log.assert_has_calls(expected_calls)
    mock_now.return_value += 300
    image_import._status_callback(action, 32, 128)
    expected_calls.append(log_call)
    mock_log.assert_has_calls(expected_calls)
    mock_now.return_value += 150
    image_import._status_callback(action, 32, 128)
    mock_log.assert_has_calls(expected_calls)
    mock_now.return_value += 3600
    image_import._status_callback(action, 32, 128)
    expected_calls.append(log_call)
    mock_log.assert_has_calls(expected_calls)