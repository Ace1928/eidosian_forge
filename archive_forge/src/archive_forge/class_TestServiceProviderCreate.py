import copy
from openstackclient.identity.v3 import service_provider
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as service_fakes
class TestServiceProviderCreate(TestServiceProvider):
    columns = ('auth_url', 'description', 'enabled', 'id', 'sp_url')
    datalist = (service_fakes.sp_auth_url, service_fakes.sp_description, True, service_fakes.sp_id, service_fakes.service_provider_url)

    def setUp(self):
        super(TestServiceProviderCreate, self).setUp()
        copied_sp = copy.deepcopy(service_fakes.SERVICE_PROVIDER)
        resource = fakes.FakeResource(None, copied_sp, loaded=True)
        self.service_providers_mock.create.return_value = resource
        self.cmd = service_provider.CreateServiceProvider(self.app, None)

    def test_create_service_provider_required_options_only(self):
        arglist = ['--auth-url', service_fakes.sp_auth_url, '--service-provider-url', service_fakes.service_provider_url, service_fakes.sp_id]
        verifylist = [('auth_url', service_fakes.sp_auth_url), ('service_provider_url', service_fakes.service_provider_url), ('service_provider_id', service_fakes.sp_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'enabled': True, 'description': None, 'auth_url': service_fakes.sp_auth_url, 'sp_url': service_fakes.service_provider_url}
        self.service_providers_mock.create.assert_called_with(id=service_fakes.sp_id, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, data)

    def test_create_service_provider_description(self):
        arglist = ['--description', service_fakes.sp_description, '--auth-url', service_fakes.sp_auth_url, '--service-provider-url', service_fakes.service_provider_url, service_fakes.sp_id]
        verifylist = [('description', service_fakes.sp_description), ('auth_url', service_fakes.sp_auth_url), ('service_provider_url', service_fakes.service_provider_url), ('service_provider_id', service_fakes.sp_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'description': service_fakes.sp_description, 'auth_url': service_fakes.sp_auth_url, 'sp_url': service_fakes.service_provider_url, 'enabled': True}
        self.service_providers_mock.create.assert_called_with(id=service_fakes.sp_id, **kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, data)

    def test_create_service_provider_disabled(self):
        service_provider = copy.deepcopy(service_fakes.SERVICE_PROVIDER)
        service_provider['enabled'] = False
        service_provider['description'] = None
        resource = fakes.FakeResource(None, service_provider, loaded=True)
        self.service_providers_mock.create.return_value = resource
        arglist = ['--auth-url', service_fakes.sp_auth_url, '--service-provider-url', service_fakes.service_provider_url, '--disable', service_fakes.sp_id]
        verifylist = [('auth_url', service_fakes.sp_auth_url), ('service_provider_url', service_fakes.service_provider_url), ('service_provider_id', service_fakes.sp_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'auth_url': service_fakes.sp_auth_url, 'sp_url': service_fakes.service_provider_url, 'enabled': False, 'description': None}
        self.service_providers_mock.create.assert_called_with(id=service_fakes.sp_id, **kwargs)
        self.assertEqual(self.columns, columns)
        datalist = (service_fakes.sp_auth_url, None, False, service_fakes.sp_id, service_fakes.service_provider_url)
        self.assertEqual(datalist, data)