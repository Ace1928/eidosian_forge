from unittest import mock
from openstackclient.network.v2 import (
from openstackclient.tests.unit.network.v2 import fakes
class TestListNetworkServiceProvider(TestNetworkServiceProvider):
    provider_list = fakes.FakeNetworkServiceProvider.create_network_service_providers(count=2)
    columns = ('Service Type', 'Name', 'Default')
    data = []
    for provider in provider_list:
        data.append((provider.service_type, provider.name, provider.is_default))

    def setUp(self):
        super(TestListNetworkServiceProvider, self).setUp()
        self.network_client.service_providers = mock.Mock(return_value=self.provider_list)
        self.cmd = service_provider.ListNetworkServiceProvider(self.app, self.namespace)

    def test_network_service_provider_list(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.service_providers.assert_called_with()
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))