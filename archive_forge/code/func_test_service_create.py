from keystoneclient import exceptions as identity_exc
from osc_lib import exceptions
from openstackclient.identity.v2_0 import service
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
def test_service_create(self):
    arglist = [self.fake_service_c.type]
    verifylist = [('type', self.fake_service_c.type), ('name', None), ('description', None)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.services_mock.create.assert_called_with(None, self.fake_service_c.type, None)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.datalist, data)