from keystoneclient import exceptions as identity_exc
from osc_lib import exceptions
from openstackclient.identity.v2_0 import service
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
def test_service_create_description(self):
    arglist = ['--name', self.fake_service_c.name, '--description', self.fake_service_c.description, self.fake_service_c.type]
    verifylist = [('type', self.fake_service_c.type), ('name', self.fake_service_c.name), ('description', self.fake_service_c.description)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.services_mock.create.assert_called_with(self.fake_service_c.name, self.fake_service_c.type, self.fake_service_c.description)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.datalist, data)