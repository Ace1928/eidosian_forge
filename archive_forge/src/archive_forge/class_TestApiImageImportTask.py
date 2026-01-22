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
class TestApiImageImportTask(test_utils.BaseTestCase):

    def setUp(self):
        super(TestApiImageImportTask, self).setUp()
        self.wd_task_input = {'import_req': {'method': {'name': 'web-download', 'uri': 'http://example.com/image.browncow'}}}
        self.gd_task_input = {'import_req': {'method': {'name': 'glance-direct'}}}
        self.mock_task_repo = mock.MagicMock()
        self.mock_image_repo = mock.MagicMock()
        self.mock_image = self.mock_image_repo.get.return_value
        self.mock_image.extra_properties = {'os_glance_import_task': TASK_ID1, 'os_glance_stage_host': 'http://glance2'}

    @mock.patch('glance.async_.flows.api_image_import._VerifyStaging.__init__')
    @mock.patch('taskflow.patterns.linear_flow.Flow.add')
    @mock.patch('taskflow.patterns.linear_flow.__init__')
    def _pass_uri(self, mock_lf_init, mock_flow_add, mock_VS_init, uri, file_uri, import_req):
        flow_kwargs = {'task_id': TASK_ID1, 'task_type': TASK_TYPE, 'task_repo': self.mock_task_repo, 'image_repo': self.mock_image_repo, 'image_id': IMAGE_ID1, 'context': mock.MagicMock(), 'import_req': import_req}
        mock_lf_init.return_value = None
        mock_VS_init.return_value = None
        self.config(node_staging_uri=uri)
        import_flow.get_flow(**flow_kwargs)
        mock_VS_init.assert_called_with(TASK_ID1, TASK_TYPE, self.mock_task_repo, file_uri)

    def test_get_flow_handles_node_uri_with_ending_slash(self):
        test_uri = 'file:///some/where/'
        expected_uri = '{0}{1}'.format(test_uri, IMAGE_ID1)
        self._pass_uri(uri=test_uri, file_uri=expected_uri, import_req=self.gd_task_input['import_req'])
        self._pass_uri(uri=test_uri, file_uri=expected_uri, import_req=self.wd_task_input['import_req'])

    def test_get_flow_handles_node_uri_without_ending_slash(self):
        test_uri = 'file:///some/where'
        expected_uri = '{0}/{1}'.format(test_uri, IMAGE_ID1)
        self._pass_uri(uri=test_uri, file_uri=expected_uri, import_req=self.wd_task_input['import_req'])
        self._pass_uri(uri=test_uri, file_uri=expected_uri, import_req=self.gd_task_input['import_req'])

    def test_get_flow_pops_stage_host(self):
        import_flow.get_flow(task_id=TASK_ID1, task_type=TASK_TYPE, task_repo=self.mock_task_repo, image_repo=self.mock_image_repo, image_id=IMAGE_ID1, context=mock.MagicMock(), import_req=self.gd_task_input['import_req'])
        self.assertNotIn('os_glance_stage_host', self.mock_image.extra_properties)
        self.assertIn('os_glance_import_task', self.mock_image.extra_properties)

    def test_assert_quota_no_task(self):
        ignored = mock.MagicMock()
        task_repo = mock.MagicMock()
        task_repo.get.return_value = None
        task_id = 'some-task'
        enforce_fn = mock.MagicMock()
        enforce_fn.side_effect = exception.LimitExceeded
        with mock.patch.object(import_flow, 'LOG') as mock_log:
            self.assertRaises(exception.LimitExceeded, import_flow.assert_quota, ignored, task_repo, task_id, [], ignored, enforce_fn)
        task_repo.get.assert_called_once_with('some-task')
        mock_log.error.assert_called_once_with('Failed to find task %r to update after quota failure', 'some-task')
        task_repo.save.assert_not_called()

    def test_assert_quota(self):
        ignored = mock.MagicMock()
        task_repo = mock.MagicMock()
        task_id = 'some-task'
        enforce_fn = mock.MagicMock()
        enforce_fn.side_effect = exception.LimitExceeded
        wrapper = mock.MagicMock()
        action = wrapper.__enter__.return_value
        action.image_status = 'importing'
        self.assertRaises(exception.LimitExceeded, import_flow.assert_quota, ignored, task_repo, task_id, ['store1'], wrapper, enforce_fn)
        action.remove_importing_stores.assert_called_once_with(['store1'])
        action.set_image_attribute.assert_called_once_with(status='queued')
        task_repo.get.assert_called_once_with('some-task')
        task_repo.save.assert_called_once_with(task_repo.get.return_value)

    def test_assert_quota_copy(self):
        ignored = mock.MagicMock()
        task_repo = mock.MagicMock()
        task_id = 'some-task'
        enforce_fn = mock.MagicMock()
        enforce_fn.side_effect = exception.LimitExceeded
        wrapper = mock.MagicMock()
        action = wrapper.__enter__.return_value
        action.image_status = 'active'
        self.assertRaises(exception.LimitExceeded, import_flow.assert_quota, ignored, task_repo, task_id, ['store1'], wrapper, enforce_fn)
        action.remove_importing_stores.assert_called_once_with(['store1'])
        action.set_image_attribute.assert_not_called()
        task_repo.get.assert_called_once_with('some-task')
        task_repo.save.assert_called_once_with(task_repo.get.return_value)