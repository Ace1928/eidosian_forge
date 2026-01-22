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
class TestImportMetadata(test_utils.BaseTestCase):

    def setUp(self):
        super(TestImportMetadata, self).setUp()
        self.config(extra_properties=[], group='glance_download_properties')
        self.wrapper = mock.MagicMock(image_id=IMAGE_ID1)
        self.context = context.RequestContext(user_id=TENANT1, project_id=TENANT1, overwrite=False)
        self.import_req = {'method': {'glance_region': 'RegionTwo', 'glance_service_interface': 'public', 'glance_image_id': IMAGE_ID1}}

    @mock.patch('urllib.request')
    @mock.patch('glance.async_.flows.api_image_import.json')
    @mock.patch('glance.async_.utils.get_glance_endpoint')
    def test_execute_return_image_size(self, mock_gge, mock_json, mock_request):
        self.config(extra_properties=['hw:numa_nodes', 'os_hash'], group='glance_download_properties')
        mock_gge.return_value = 'https://other.cloud.foo/image'
        action = self.wrapper.__enter__.return_value
        mock_json.loads.return_value = {'status': 'active', 'disk_format': 'qcow2', 'container_format': 'bare', 'hw:numa_nodes': '2', 'os_hash': 'hash', 'extra_metadata': 'hello', 'size': '12345'}
        task = import_flow._ImportMetadata(TASK_ID1, TASK_TYPE, self.context, self.wrapper, self.import_req)
        self.assertEqual(12345, task.execute())
        mock_request.Request.assert_called_once_with('https://other.cloud.foo/image/v2/images/%s' % IMAGE_ID1, headers={'X-Auth-Token': self.context.auth_token})
        mock_gge.assert_called_once_with(self.context, 'RegionTwo', 'public')
        action.set_image_attribute.assert_called_once_with(disk_format='qcow2', container_format='bare')
        action.set_image_extra_properties.assert_called_once_with({'hw:numa_nodes': '2', 'os_hash': 'hash'})

    @mock.patch('urllib.request')
    @mock.patch('glance.async_.utils.get_glance_endpoint')
    def test_execute_fail_no_glance_endpoint(self, mock_gge, mock_request):
        action = self.wrapper.__enter__.return_value
        mock_gge.side_effect = exception.GlanceEndpointNotFound(region='RegionTwo', interface='public')
        task = import_flow._ImportMetadata(TASK_ID1, TASK_TYPE, self.context, self.wrapper, self.import_req)
        self.assertRaises(exception.GlanceEndpointNotFound, task.execute)
        action.assert_not_called()
        mock_request.assert_not_called()

    @mock.patch('urllib.request')
    @mock.patch('glance.async_.utils.get_glance_endpoint')
    def test_execute_fail_remote_glance_unreachable(self, mock_gge, mock_r):
        action = self.wrapper.__enter__.return_value
        mock_r.urlopen.side_effect = urllib.error.HTTPError('/file', 400, 'Test Fail', {}, None)
        task = import_flow._ImportMetadata(TASK_ID1, TASK_TYPE, self.context, self.wrapper, self.import_req)
        self.assertRaises(urllib.error.HTTPError, task.execute)
        action.assert_not_called()

    @mock.patch('urllib.request')
    @mock.patch('glance.async_.flows.api_image_import.json')
    @mock.patch('glance.async_.utils.get_glance_endpoint')
    def test_execute_invalid_remote_image_state(self, mock_gge, mock_json, mock_request):
        action = self.wrapper.__enter__.return_value
        mock_gge.return_value = 'https://other.cloud.foo/image'
        mock_json.loads.return_value = {'status': 'queued'}
        task = import_flow._ImportMetadata(TASK_ID1, TASK_TYPE, self.context, self.wrapper, self.import_req)
        self.assertRaises(import_flow._InvalidGlanceDownloadImageStatus, task.execute)
        action.assert_not_called()

    @mock.patch('urllib.request')
    @mock.patch('glance.async_.flows.api_image_import.json')
    @mock.patch('glance.async_.utils.get_glance_endpoint')
    def test_execute_raise_if_no_size(self, mock_gge, mock_json, mock_request):
        self.config(extra_properties=['hw:numa_nodes', 'os_hash'], group='glance_download_properties')
        mock_gge.return_value = 'https://other.cloud.foo/image'
        action = self.wrapper.__enter__.return_value
        mock_json.loads.return_value = {'status': 'active', 'disk_format': 'qcow2', 'container_format': 'bare', 'hw:numa_nodes': '2', 'os_hash': 'hash', 'extra_metadata': 'hello'}
        task = import_flow._ImportMetadata(TASK_ID1, TASK_TYPE, self.context, self.wrapper, self.import_req)
        self.assertRaises(exception.ImportTaskError, task.execute)
        mock_request.Request.assert_called_once_with('https://other.cloud.foo/image/v2/images/%s' % IMAGE_ID1, headers={'X-Auth-Token': self.context.auth_token})
        mock_gge.assert_called_once_with(self.context, 'RegionTwo', 'public')
        action.set_image_attribute.assert_called_once_with(disk_format='qcow2', container_format='bare')
        action.set_image_extra_properties.assert_called_once_with({'hw:numa_nodes': '2', 'os_hash': 'hash'})

    def test_revert_rollback_metadata_value(self):
        action = self.wrapper.__enter__.return_value
        task = import_flow._ImportMetadata(TASK_ID1, TASK_TYPE, self.context, self.wrapper, self.import_req)
        task.properties = {'prop1': 'value1', 'prop2': 'value2'}
        task.old_properties = {'prop1': 'orig_val', 'old_prop': 'old_value'}
        task.old_attributes = {'container_format': 'bare', 'disk_format': 'qcow2'}
        task.revert(None)
        action.set_image_attribute.assert_called_once_with(status='queued', container_format='bare', disk_format='qcow2')
        action.pop_extra_property.assert_called_once_with('prop2')
        action.set_image_extra_properties.assert_called_once_with(task.old_properties)