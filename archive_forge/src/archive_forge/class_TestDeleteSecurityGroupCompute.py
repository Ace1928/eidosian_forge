from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import security_group
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as tests_utils
@mock.patch('openstackclient.api.compute_v2.APIv2.security_group_delete')
class TestDeleteSecurityGroupCompute(compute_fakes.TestComputev2):
    _security_groups = compute_fakes.create_security_groups()

    def setUp(self):
        super(TestDeleteSecurityGroupCompute, self).setUp()
        self.app.client_manager.network_endpoint_enabled = False
        self.compute_client.api.security_group_find = compute_fakes.get_security_groups(self._security_groups)
        self.cmd = security_group.DeleteSecurityGroup(self.app, None)

    def test_security_group_delete(self, sg_mock):
        sg_mock.return_value = mock.Mock(return_value=None)
        arglist = [self._security_groups[0]['id']]
        verifylist = [('group', [self._security_groups[0]['id']])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        sg_mock.assert_called_once_with(self._security_groups[0]['id'])
        self.assertIsNone(result)

    def test_security_group_multi_delete(self, sg_mock):
        sg_mock.return_value = mock.Mock(return_value=None)
        arglist = []
        verifylist = []
        for s in self._security_groups:
            arglist.append(s['id'])
        verifylist = [('group', arglist)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        calls = []
        for s in self._security_groups:
            calls.append(call(s['id']))
        sg_mock.assert_has_calls(calls)
        self.assertIsNone(result)

    def test_security_group_multi_delete_with_exception(self, sg_mock):
        sg_mock.return_value = mock.Mock(return_value=None)
        sg_mock.side_effect = [mock.Mock(return_value=None), exceptions.CommandError]
        arglist = [self._security_groups[0]['id'], 'unexist_security_group']
        verifylist = [('group', [self._security_groups[0]['id'], 'unexist_security_group'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        try:
            self.cmd.take_action(parsed_args)
            self.fail('CommandError should be raised.')
        except exceptions.CommandError as e:
            self.assertEqual('1 of 2 groups failed to delete.', str(e))
        sg_mock.assert_any_call(self._security_groups[0]['id'])
        sg_mock.assert_any_call('unexist_security_group')