import copy
from openstackclient.identity.v3 import service_provider
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as service_fakes
class TestServiceProviderList(TestServiceProvider):

    def setUp(self):
        super(TestServiceProviderList, self).setUp()
        self.service_providers_mock.get.return_value = fakes.FakeResource(None, copy.deepcopy(service_fakes.SERVICE_PROVIDER), loaded=True)
        self.service_providers_mock.list.return_value = [fakes.FakeResource(None, copy.deepcopy(service_fakes.SERVICE_PROVIDER), loaded=True)]
        self.cmd = service_provider.ListServiceProvider(self.app, None)

    def test_service_provider_list_no_options(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.service_providers_mock.list.assert_called_with()
        collist = ('ID', 'Enabled', 'Description', 'Auth URL')
        self.assertEqual(collist, columns)
        datalist = ((service_fakes.sp_id, True, service_fakes.sp_description, service_fakes.sp_auth_url),)
        self.assertEqual(tuple(data), datalist)