from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import security_group
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as tests_utils
@mock.patch('openstackclient.api.compute_v2.APIv2.security_group_set')
class TestSetSecurityGroupCompute(compute_fakes.TestComputev2):
    _security_group = compute_fakes.create_one_security_group()

    def setUp(self):
        super(TestSetSecurityGroupCompute, self).setUp()
        self.app.client_manager.network_endpoint_enabled = False
        self.compute_client.api.security_group_find = mock.Mock(return_value=self._security_group)
        self.cmd = security_group.SetSecurityGroup(self.app, None)

    def test_security_group_set_no_options(self, sg_mock):
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, [], [])

    def test_security_group_set_no_updates(self, sg_mock):
        sg_mock.return_value = mock.Mock(return_value=None)
        arglist = [self._security_group['name']]
        verifylist = [('group', self._security_group['name'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        sg_mock.assert_called_once_with(self._security_group, self._security_group['name'], self._security_group['description'])
        self.assertIsNone(result)

    def test_security_group_set_all_options(self, sg_mock):
        sg_mock.return_value = mock.Mock(return_value=None)
        new_name = 'new-' + self._security_group['name']
        new_description = 'new-' + self._security_group['description']
        arglist = ['--name', new_name, '--description', new_description, self._security_group['name']]
        verifylist = [('description', new_description), ('group', self._security_group['name']), ('name', new_name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        sg_mock.assert_called_once_with(self._security_group, new_name, new_description)
        self.assertIsNone(result)