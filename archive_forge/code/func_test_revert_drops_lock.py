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
@mock.patch('glance.async_.flows.api_image_import.LOG')
def test_revert_drops_lock(self, mock_log):
    wrapper = import_flow.ImportActionWrapper(self.img_repo, IMAGE_ID1, TASK_ID1)
    imagelock = import_flow._ImageLock(TASK_ID1, TASK_TYPE, wrapper)
    with mock.patch.object(wrapper, 'drop_lock_for_task') as mock_drop:
        imagelock.revert(None)
        mock_drop.assert_called_once_with()
    mock_log.debug.assert_called_once_with('Image %(image)s import task %(task)s dropped its lock after failure', {'image': IMAGE_ID1, 'task': TASK_ID1})