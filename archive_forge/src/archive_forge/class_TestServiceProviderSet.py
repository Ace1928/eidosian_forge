import copy
from openstackclient.identity.v3 import service_provider
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as service_fakes
class TestServiceProviderSet(TestServiceProvider):
    columns = ('auth_url', 'description', 'enabled', 'id', 'sp_url')
    datalist = (service_fakes.sp_auth_url, service_fakes.sp_description, False, service_fakes.sp_id, service_fakes.service_provider_url)

    def setUp(self):
        super(TestServiceProviderSet, self).setUp()
        self.cmd = service_provider.SetServiceProvider(self.app, None)

    def test_service_provider_disable(self):
        """Disable Service Provider

        Set Service Provider's ``enabled`` attribute to False.
        """

        def prepare(self):
            """Prepare fake return objects before the test is executed"""
            updated_sp = copy.deepcopy(service_fakes.SERVICE_PROVIDER)
            updated_sp['enabled'] = False
            resources = fakes.FakeResource(None, updated_sp, loaded=True)
            self.service_providers_mock.update.return_value = resources
        prepare(self)
        arglist = ['--disable', service_fakes.sp_id]
        verifylist = [('service_provider', service_fakes.sp_id), ('enable', False), ('disable', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.service_providers_mock.update.assert_called_with(service_fakes.sp_id, enabled=False, description=None, auth_url=None, sp_url=None)

    def test_service_provider_enable(self):
        """Enable Service Provider.

        Set Service Provider's ``enabled`` attribute to True.
        """

        def prepare(self):
            """Prepare fake return objects before the test is executed"""
            resources = fakes.FakeResource(None, copy.deepcopy(service_fakes.SERVICE_PROVIDER), loaded=True)
            self.service_providers_mock.update.return_value = resources
        prepare(self)
        arglist = ['--enable', service_fakes.sp_id]
        verifylist = [('service_provider', service_fakes.sp_id), ('enable', True), ('disable', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.service_providers_mock.update.assert_called_with(service_fakes.sp_id, enabled=True, description=None, auth_url=None, sp_url=None)

    def test_service_provider_no_options(self):

        def prepare(self):
            """Prepare fake return objects before the test is executed"""
            resources = fakes.FakeResource(None, copy.deepcopy(service_fakes.SERVICE_PROVIDER), loaded=True)
            self.service_providers_mock.get.return_value = resources
            resources = fakes.FakeResource(None, copy.deepcopy(service_fakes.SERVICE_PROVIDER), loaded=True)
            self.service_providers_mock.update.return_value = resources
        prepare(self)
        arglist = [service_fakes.sp_id]
        verifylist = [('service_provider', service_fakes.sp_id), ('description', None), ('enable', False), ('disable', False), ('auth_url', None), ('service_provider_url', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)