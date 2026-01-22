import os
import sys
import tempfile
from unittest import mock
import uuid
import fixtures
import io
from keystoneauth1 import fixture as keystone_fixture
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from requests_mock.contrib import fixture as rm_fixture
import testscenarios
import testtools
from urllib import parse
from urllib import request
import yaml
from heatclient._i18n import _
from heatclient.common import http
from heatclient.common import utils
from heatclient import exc
import heatclient.shell
from heatclient.tests.unit import fakes
import heatclient.v1.shell
class ShellTestDeployment(ShellBase):

    def setUp(self):
        super(ShellTestDeployment, self).setUp()
        self.client = http.SessionClient
        self._set_fake_env()

    def _set_fake_env(self):
        """Patch os.environ to avoid required auth info."""
        self.set_fake_env(FAKE_ENV_KEYSTONE_V2)

    def test_deploy_create(self):
        self.register_keystone_auth_fixture()
        self.patch('heatclient.common.deployment_utils.build_derived_config_params')
        self.patch('heatclient.common.deployment_utils.build_signal_id')
        resp_dict = {'software_deployment': {'status': 'INPROGRESS', 'server_id': '700115e5-0100-4ecc-9ef7-9e05f27d8803', 'config_id': '18c4fc03-f897-4a1d-aaad-2b7622e60257', 'output_values': {'deploy_stdout': '', 'deploy_stderr': '', 'deploy_status_code': 0, 'result': 'The result value'}, 'input_values': {}, 'action': 'UPDATE', 'status_reason': 'Outputs received', 'id': 'abcd'}}
        config_dict = {'software_config': {'inputs': [], 'group': 'script', 'name': 'config_name', 'outputs': [], 'options': {}, 'config': 'the config script', 'id': 'defg'}}
        derived_dict = {'software_config': {'inputs': [], 'group': 'script', 'name': 'config_name', 'outputs': [], 'options': {}, 'config': 'the config script', 'id': 'abcd'}}
        deploy_data = {'action': 'UPDATE', 'config_id': 'abcd', 'server_id': 'inst01', 'status': 'IN_PROGRESS', 'tenant_id': 'asdf'}
        self.mock_request_get('/software_configs/defg', config_dict)
        self.mock_request_post('/software_configs', derived_dict, data={})
        self.mock_request_post('/software_deployments', resp_dict, data=deploy_data)
        self.mock_request_post('/software_configs', derived_dict, data={})
        self.mock_request_post('/software_deployments', resp_dict, data=deploy_data)
        self.mock_request_error('/software_configs/defgh', 'GET', exc.HTTPNotFound())
        text = self.shell('deployment-create -c defg -sinst01 xxx')
        required = ['status', 'server_id', 'config_id', 'output_values', 'input_values', 'action', 'status_reason', 'id']
        for r in required:
            self.assertRegex(text, r)
        text = self.shell('deployment-create -sinst01 xxx')
        for r in required:
            self.assertRegex(text, r)
        self.assertRaises(exc.CommandError, self.shell, 'deployment-create -c defgh -s inst01 yyy')

    def test_deploy_list(self):
        self.register_keystone_auth_fixture()
        resp_dict = {'software_deployments': [{'status': 'COMPLETE', 'server_id': '123', 'config_id': '18c4fc03-f897-4a1d-aaad-2b7622e60257', 'output_values': {'deploy_stdout': '', 'deploy_stderr': '', 'deploy_status_code': 0, 'result': 'The result value'}, 'input_values': {}, 'action': 'CREATE', 'status_reason': 'Outputs received', 'id': 'defg'}]}
        self.mock_request_get('/software_deployments?', resp_dict)
        self.mock_request_get('/software_deployments?server_id=123', resp_dict)
        list_text = self.shell('deployment-list')
        required = ['id', 'config_id', 'server_id', 'action', 'status', 'creation_time', 'status_reason']
        for r in required:
            self.assertRegex(list_text, r)
        self.assertNotRegex(list_text, 'parent')
        list_text = self.shell('deployment-list -s 123')
        for r in required:
            self.assertRegex(list_text, r)
        self.assertNotRegex(list_text, 'parent')

    def test_deploy_show(self):
        self.register_keystone_auth_fixture()
        resp_dict = {'software_deployment': {'status': 'COMPLETE', 'server_id': '700115e5-0100-4ecc-9ef7-9e05f27d8803', 'config_id': '18c4fc03-f897-4a1d-aaad-2b7622e60257', 'output_values': {'deploy_stdout': '', 'deploy_stderr': '', 'deploy_status_code': 0, 'result': 'The result value'}, 'input_values': {}, 'action': 'CREATE', 'status_reason': 'Outputs received', 'id': 'defg'}}
        self.mock_request_get('/software_deployments/defg', resp_dict)
        self.mock_request_error('/software_deployments/defgh', 'GET', exc.HTTPNotFound())
        text = self.shell('deployment-show defg')
        required = ['status', 'server_id', 'config_id', 'output_values', 'input_values', 'action', 'status_reason', 'id']
        for r in required:
            self.assertRegex(text, r)
        self.assertRaises(exc.CommandError, self.shell, 'deployment-show defgh')

    def test_deploy_delete(self):
        self.register_keystone_auth_fixture()
        deploy_resp_dict = {'software_deployment': {'config_id': 'dummy_config_id'}}

        def _get_deployment_request_except(id):
            self.mock_request_error('/software_deployments/%s' % id, 'GET', exc.HTTPNotFound())

        def _delete_deployment_request_except(id):
            self.mock_request_get('/software_deployments/%s' % id, deploy_resp_dict)
            self.mock_request_error('/software_deployments/%s' % id, 'DELETE', exc.HTTPNotFound())

        def _delete_config_request_except(id):
            self.mock_request_get('/software_deployments/%s' % id, deploy_resp_dict)
            self.mock_request_delete('/software_deployments/%s' % id)
            self.mock_request_error('/software_configs/dummy_config_id', 'DELETE', exc.HTTPNotFound())

        def _delete_request_success(id):
            self.mock_request_get('/software_deployments/%s' % id, deploy_resp_dict)
            self.mock_request_delete('/software_deployments/%s' % id)
            self.mock_request_delete('/software_configs/dummy_config_id')
        _get_deployment_request_except('defg')
        _get_deployment_request_except('qwer')
        _delete_deployment_request_except('defg')
        _delete_deployment_request_except('qwer')
        _delete_config_request_except('defg')
        _delete_config_request_except('qwer')
        _delete_request_success('defg')
        _delete_request_success('qwer')
        error = self.assertRaises(exc.CommandError, self.shell, 'deployment-delete defg qwer')
        self.assertIn('Unable to delete 2 of the 2 deployments.', str(error))
        error2 = self.assertRaises(exc.CommandError, self.shell, 'deployment-delete defg qwer')
        self.assertIn('Unable to delete 2 of the 2 deployments.', str(error2))
        output = self.shell('deployment-delete defg qwer')
        self.assertRegex(output, 'Failed to delete the correlative config dummy_config_id of deployment defg')
        self.assertRegex(output, 'Failed to delete the correlative config dummy_config_id of deployment qwer')
        self.assertEqual('', self.shell('deployment-delete defg qwer'))

    def test_deploy_metadata(self):
        self.register_keystone_auth_fixture()
        resp_dict = {'metadata': [{'id': 'abcd'}, {'id': 'defg'}]}
        self.mock_request_get('/software_deployments/metadata/aaaa', resp_dict)
        build_info_text = self.shell('deployment-metadata-show aaaa')
        required = ['abcd', 'defg', 'id']
        for r in required:
            self.assertRegex(build_info_text, r)

    def test_deploy_output_show(self):
        self.register_keystone_auth_fixture()
        resp_dict = {'software_deployment': {'status': 'COMPLETE', 'server_id': '700115e5-0100-4ecc-9ef7-9e05f27d8803', 'config_id': '18c4fc03-f897-4a1d-aaad-2b7622e60257', 'output_values': {'deploy_stdout': '', 'deploy_stderr': '', 'deploy_status_code': 0, 'result': 'The result value', 'dict_output': {'foo': 'bar'}, 'list_output': ['foo', 'bar']}, 'input_values': {}, 'action': 'CREATE', 'status_reason': 'Outputs received', 'id': 'defg'}}
        self.mock_request_error('/software_deployments/defgh', 'GET', exc.HTTPNotFound())
        for a in range(9):
            self.mock_request_get('/software_deployments/defg', resp_dict)
        self.assertRaises(exc.CommandError, self.shell, 'deployment-output-show defgh result')
        self.assertEqual('The result value\n', self.shell('deployment-output-show defg result'))
        self.assertEqual('"The result value"\n', self.shell('deployment-output-show --format json defg result'))
        self.assertEqual('{\n  "foo": "bar"\n}\n', self.shell('deployment-output-show defg dict_output'))
        self.assertEqual(self.shell('deployment-output-show --format raw defg dict_output'), self.shell('deployment-output-show --format json defg dict_output'))
        self.assertEqual('[\n  "foo", \n  "bar"\n]\n', self.shell('deployment-output-show defg list_output'))
        self.assertEqual(self.shell('deployment-output-show --format raw defg list_output'), self.shell('deployment-output-show --format json defg list_output'))
        self.assertEqual({'deploy_stdout': '', 'deploy_stderr': '', 'deploy_status_code': 0, 'result': 'The result value', 'dict_output': {'foo': 'bar'}, 'list_output': ['foo', 'bar']}, jsonutils.loads(self.shell('deployment-output-show --format json defg --all')))