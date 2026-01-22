import copy
from unittest import mock
from osc_lib import exceptions as exc
from heatclient import exc as heat_exc
from heatclient.osc.v1 import software_deployment
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
from heatclient.v1 import software_configs
from heatclient.v1 import software_deployments
class TestDeploymentCreate(TestDeployment):
    server_id = '1234'
    config_id = '5678'
    deploy_id = '910'
    config = {'name': 'my_deploy', 'group': 'strict', 'config': '#!/bin/bash', 'inputs': [], 'outputs': [], 'options': [], 'id': config_id}
    deployment = {'server_id': server_id, 'input_values': {}, 'action': 'UPDATE', 'status': 'IN_PROGRESS', 'status_reason': None, 'signal_id': 'signal_id', 'config_id': config_id, 'id': deploy_id}
    config_defaults = {'group': 'Heat::Ungrouped', 'config': '', 'options': {}, 'inputs': [{'name': 'deploy_server_id', 'description': 'ID of the server being deployed to', 'type': 'String', 'value': server_id}, {'name': 'deploy_action', 'description': 'Name of the current action being deployed', 'type': 'String', 'value': 'UPDATE'}, {'name': 'deploy_signal_transport', 'description': 'How the server should signal to heat with the deployment output values.', 'type': 'String', 'value': 'TEMP_URL_SIGNAL'}, {'name': 'deploy_signal_id', 'description': 'ID of signal to use for signaling output values', 'type': 'String', 'value': 'signal_id'}, {'name': 'deploy_signal_verb', 'description': 'HTTP verb to use for signaling output values', 'type': 'String', 'value': 'PUT'}], 'outputs': [], 'name': 'my_deploy'}
    deploy_defaults = {'config_id': config_id, 'server_id': server_id, 'action': 'UPDATE', 'status': 'IN_PROGRESS'}

    def setUp(self):
        super(TestDeploymentCreate, self).setUp()
        self.cmd = software_deployment.CreateDeployment(self.app, None)
        self.config_client.create.return_value = software_configs.SoftwareConfig(None, self.config)
        self.config_client.get.return_value = software_configs.SoftwareConfig(None, self.config)
        self.sd_client.create.return_value = software_deployments.SoftwareDeployment(None, self.deployment)

    @mock.patch('heatclient.common.deployment_utils.build_signal_id', return_value='signal_id')
    def test_deployment_create(self, mock_build):
        arglist = ['my_deploy', '--server', self.server_id]
        expected_cols = ('action', 'config_id', 'id', 'input_values', 'server_id', 'signal_id', 'status', 'status_reason')
        expected_data = ('UPDATE', self.config_id, self.deploy_id, {}, self.server_id, 'signal_id', 'IN_PROGRESS', None)
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.config_client.create.assert_called_with(**self.config_defaults)
        self.sd_client.create.assert_called_with(**self.deploy_defaults)
        self.assertEqual(expected_cols, columns)
        self.assertEqual(expected_data, data)

    @mock.patch('heatclient.common.deployment_utils.build_signal_id', return_value='signal_id')
    def test_deployment_create_with_config(self, mock_build):
        arglist = ['my_deploy', '--server', self.server_id, '--config', self.config_id]
        config = copy.deepcopy(self.config_defaults)
        config['config'] = '#!/bin/bash'
        config['group'] = 'strict'
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.config_client.get.assert_called_with(self.config_id)
        self.config_client.create.assert_called_with(**config)
        self.sd_client.create.assert_called_with(**self.deploy_defaults)

    def test_deployment_create_config_not_found(self):
        arglist = ['my_deploy', '--server', self.server_id, '--config', 'bad_id']
        self.config_client.get.side_effect = heat_exc.HTTPNotFound
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)

    def test_deployment_create_no_signal(self):
        arglist = ['my_deploy', '--server', self.server_id, '--signal-transport', 'NO_SIGNAL']
        config = copy.deepcopy(self.config_defaults)
        config['inputs'] = config['inputs'][:-2]
        config['inputs'][2]['value'] = 'NO_SIGNAL'
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.config_client.create.assert_called_with(**config)
        self.sd_client.create.assert_called_with(**self.deploy_defaults)

    @mock.patch('heatclient.common.deployment_utils.build_signal_id', return_value='signal_id')
    def test_deployment_create_invalid_signal_transport(self, mock_build):
        arglist = ['my_deploy', '--server', self.server_id, '--signal-transport', 'A']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.assertRaises(heat_exc.CommandError, self.cmd.take_action, parsed_args)

    @mock.patch('heatclient.common.deployment_utils.build_signal_id', return_value='signal_id')
    def test_deployment_create_input_value(self, mock_build):
        arglist = ['my_deploy', '--server', self.server_id, '--input-value', 'foo=bar']
        config = copy.deepcopy(self.config_defaults)
        config['inputs'].insert(0, {'name': 'foo', 'type': 'String', 'value': 'bar'})
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.config_client.create.assert_called_with(**config)
        self.sd_client.create.assert_called_with(**self.deploy_defaults)

    @mock.patch('heatclient.common.deployment_utils.build_signal_id', return_value='signal_id')
    def test_deployment_create_action(self, mock_build):
        arglist = ['my_deploy', '--server', self.server_id, '--action', 'DELETE']
        config = copy.deepcopy(self.config_defaults)
        config['inputs'][1]['value'] = 'DELETE'
        deploy = copy.deepcopy(self.deploy_defaults)
        deploy['action'] = 'DELETE'
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.config_client.create.assert_called_with(**config)
        self.sd_client.create.assert_called_with(**deploy)