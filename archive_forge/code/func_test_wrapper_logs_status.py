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
@mock.patch.object(import_flow, 'LOG')
def test_wrapper_logs_status(self, mock_log):
    mock_repo = mock.MagicMock()
    mock_image = mock_repo.get.return_value
    mock_image.extra_properties = {'os_glance_import_task': TASK_ID1}
    wrapper = import_flow.ImportActionWrapper(mock_repo, IMAGE_ID1, TASK_ID1)
    mock_image.status = 'foo'
    with wrapper as action:
        action.set_image_attribute(status='bar')
    mock_log.debug.assert_called_once_with('Image %(image_id)s status changing from %(old_status)s to %(new_status)s', {'image_id': IMAGE_ID1, 'old_status': 'foo', 'new_status': 'bar'})
    self.assertEqual('bar', mock_image.status)