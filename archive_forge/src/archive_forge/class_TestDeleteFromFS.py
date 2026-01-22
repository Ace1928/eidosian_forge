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
class TestDeleteFromFS(test_utils.BaseTestCase):

    def test_delete_with_backends_deletes(self):
        task = import_flow._DeleteFromFS(TASK_ID1, TASK_TYPE)
        self.config(enabled_backends='file:foo')
        with mock.patch.object(import_flow.store_api, 'delete') as mock_del:
            task.execute(mock.sentinel.path)
            mock_del.assert_called_once_with(mock.sentinel.path, 'os_glance_staging_store')

    def test_delete_with_backends_delete_fails(self):
        self.config(enabled_backends='file:foo')
        task = import_flow._DeleteFromFS(TASK_ID1, TASK_TYPE)
        with mock.patch.object(import_flow.store_api, 'delete') as mock_del:
            mock_del.side_effect = store_exceptions.NotFound(image=IMAGE_ID1, message='Testing')
            task.execute(mock.sentinel.path)
            mock_del.assert_called_once_with(mock.sentinel.path, 'os_glance_staging_store')
            mock_del.side_effect = RuntimeError
            self.assertRaises(RuntimeError, task.execute, mock.sentinel.path)

    @mock.patch('os.path.exists')
    @mock.patch('os.unlink')
    def test_delete_without_backends_exists(self, mock_unlink, mock_exists):
        mock_exists.return_value = True
        task = import_flow._DeleteFromFS(TASK_ID1, TASK_TYPE)
        task.execute('1234567foo')
        mock_unlink.assert_called_once_with('foo')
        mock_unlink.reset_mock()
        mock_unlink.side_effect = OSError(123, 'failed')
        task.execute('1234567foo')

    @mock.patch('os.path.exists')
    @mock.patch('os.unlink')
    def test_delete_without_backends_missing(self, mock_unlink, mock_exists):
        mock_exists.return_value = False
        task = import_flow._DeleteFromFS(TASK_ID1, TASK_TYPE)
        task.execute('foo')
        mock_unlink.assert_not_called()