from unittest import mock
from osc_lib import exceptions
from openstackclient.network.v2 import network_flavor_profile
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
class TestDeleteFlavorProfile(TestFlavorProfile):
    _network_flavor_profiles = network_fakes.create_service_profile(count=2)

    def setUp(self):
        super(TestDeleteFlavorProfile, self).setUp()
        self.network_client.delete_service_profile = mock.Mock(return_value=None)
        self.network_client.find_service_profile = network_fakes.get_service_profile(flavor_profile=self._network_flavor_profiles)
        self.cmd = network_flavor_profile.DeleteNetworkFlavorProfile(self.app, self.namespace)

    def test_network_flavor_profile_delete(self):
        arglist = [self._network_flavor_profiles[0].id]
        verifylist = [('flavor_profile', [self._network_flavor_profiles[0].id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.find_service_profile.assert_called_once_with(self._network_flavor_profiles[0].id, ignore_missing=False)
        self.network_client.delete_service_profile.assert_called_once_with(self._network_flavor_profiles[0])
        self.assertIsNone(result)

    def test_multi_network_flavor_profiles_delete(self):
        arglist = []
        for a in self._network_flavor_profiles:
            arglist.append(a.id)
        verifylist = [('flavor_profile', arglist)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        calls = []
        for a in self._network_flavor_profiles:
            calls.append(mock.call(a))
        self.network_client.delete_service_profile.assert_has_calls(calls)
        self.assertIsNone(result)

    def test_multi_network_flavor_profiles_delete_with_exception(self):
        arglist = [self._network_flavor_profiles[0].id, 'unexist_network_flavor_profile']
        verifylist = [('flavor_profile', [self._network_flavor_profiles[0].id, 'unexist_network_flavor_profile'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        find_mock_result = [self._network_flavor_profiles[0], exceptions.CommandError]
        self.network_client.find_service_profile = mock.Mock(side_effect=find_mock_result)
        try:
            self.cmd.take_action(parsed_args)
            self.fail('CommandError should be raised.')
        except exceptions.CommandError as e:
            self.assertEqual('1 of 2 flavor_profiles failed to delete.', str(e))
        self.network_client.find_service_profile.assert_any_call(self._network_flavor_profiles[0].id, ignore_missing=False)
        self.network_client.find_service_profile.assert_any_call('unexist_network_flavor_profile', ignore_missing=False)
        self.network_client.delete_service_profile.assert_called_once_with(self._network_flavor_profiles[0])