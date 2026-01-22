from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import security_group
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestDeleteSecurityGroupNetwork(TestSecurityGroupNetwork):
    _security_groups = network_fakes.FakeSecurityGroup.create_security_groups()

    def setUp(self):
        super(TestDeleteSecurityGroupNetwork, self).setUp()
        self.network_client.delete_security_group = mock.Mock(return_value=None)
        self.network_client.find_security_group = network_fakes.FakeSecurityGroup.get_security_groups(self._security_groups)
        self.cmd = security_group.DeleteSecurityGroup(self.app, self.namespace)

    def test_security_group_delete(self):
        arglist = [self._security_groups[0].name]
        verifylist = [('group', [self._security_groups[0].name])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.delete_security_group.assert_called_once_with(self._security_groups[0])
        self.assertIsNone(result)

    def test_multi_security_groups_delete(self):
        arglist = []
        verifylist = []
        for s in self._security_groups:
            arglist.append(s.name)
        verifylist = [('group', arglist)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        calls = []
        for s in self._security_groups:
            calls.append(call(s))
        self.network_client.delete_security_group.assert_has_calls(calls)
        self.assertIsNone(result)

    def test_multi_security_groups_delete_with_exception(self):
        arglist = [self._security_groups[0].name, 'unexist_security_group']
        verifylist = [('group', [self._security_groups[0].name, 'unexist_security_group'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        find_mock_result = [self._security_groups[0], exceptions.CommandError]
        self.network_client.find_security_group = mock.Mock(side_effect=find_mock_result)
        try:
            self.cmd.take_action(parsed_args)
            self.fail('CommandError should be raised.')
        except exceptions.CommandError as e:
            self.assertEqual('1 of 2 groups failed to delete.', str(e))
        self.network_client.find_security_group.assert_any_call(self._security_groups[0].name, ignore_missing=False)
        self.network_client.find_security_group.assert_any_call('unexist_security_group', ignore_missing=False)
        self.network_client.delete_security_group.assert_called_once_with(self._security_groups[0])