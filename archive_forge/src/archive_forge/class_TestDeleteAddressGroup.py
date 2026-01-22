from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import address_group
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestDeleteAddressGroup(TestAddressGroup):
    _address_groups = network_fakes.create_address_groups(count=2)

    def setUp(self):
        super(TestDeleteAddressGroup, self).setUp()
        self.network_client.delete_address_group = mock.Mock(return_value=None)
        self.network_client.find_address_group = network_fakes.get_address_groups(address_groups=self._address_groups)
        self.cmd = address_group.DeleteAddressGroup(self.app, self.namespace)

    def test_address_group_delete(self):
        arglist = [self._address_groups[0].name]
        verifylist = [('address_group', [self._address_groups[0].name])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.find_address_group.assert_called_once_with(self._address_groups[0].name, ignore_missing=False)
        self.network_client.delete_address_group.assert_called_once_with(self._address_groups[0])
        self.assertIsNone(result)

    def test_multi_address_groups_delete(self):
        arglist = []
        for a in self._address_groups:
            arglist.append(a.name)
        verifylist = [('address_group', arglist)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        calls = []
        for a in self._address_groups:
            calls.append(call(a))
        self.network_client.delete_address_group.assert_has_calls(calls)
        self.assertIsNone(result)

    def test_multi_address_groups_delete_with_exception(self):
        arglist = [self._address_groups[0].name, 'unexist_address_group']
        verifylist = [('address_group', [self._address_groups[0].name, 'unexist_address_group'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        find_mock_result = [self._address_groups[0], exceptions.CommandError]
        self.network_client.find_address_group = mock.Mock(side_effect=find_mock_result)
        try:
            self.cmd.take_action(parsed_args)
            self.fail('CommandError should be raised.')
        except exceptions.CommandError as e:
            self.assertEqual('1 of 2 address groups failed to delete.', str(e))
        self.network_client.find_address_group.assert_any_call(self._address_groups[0].name, ignore_missing=False)
        self.network_client.find_address_group.assert_any_call('unexist_address_group', ignore_missing=False)
        self.network_client.delete_address_group.assert_called_once_with(self._address_groups[0])