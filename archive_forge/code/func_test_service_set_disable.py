from keystoneclient import exceptions as identity_exc
from osc_lib import exceptions
from openstackclient.identity.v3 import service
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_service_set_disable(self):
    arglist = ['--disable', self.service.name]
    verifylist = [('type', None), ('name', None), ('description', None), ('enable', False), ('disable', True), ('service', self.service.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    kwargs = {'enabled': False}
    self.services_mock.update.assert_called_with(self.service.id, **kwargs)
    self.assertIsNone(result)