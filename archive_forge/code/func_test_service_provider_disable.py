import copy
from openstackclient.identity.v3 import service_provider
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as service_fakes
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