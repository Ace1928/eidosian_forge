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
@mock.patch('glance.common.scripts.utils.get_task')
def test_status_aborts_invalid_task_state(self, mock_get):
    task_repo = mock.MagicMock()
    image_import = import_flow._ImportToStore(TASK_ID1, TASK_TYPE, task_repo, mock.MagicMock(), 'http://url', 'store1', False, True)
    task = mock.MagicMock()
    task.status = 'failed'
    mock_get.return_value = task
    action = mock.MagicMock()
    self.assertRaises(exception.TaskAbortedError, image_import._status_callback, action, 128, 256 * units.Mi)
    mock_get.assert_called_once_with(task_repo, TASK_ID1)
    task_repo.save.assert_not_called()