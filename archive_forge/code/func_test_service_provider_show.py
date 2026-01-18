import copy
from openstackclient.identity.v3 import service_provider
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as service_fakes
def test_service_provider_show(self):
    arglist = [service_fakes.sp_id]
    verifylist = [('service_provider', service_fakes.sp_id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.service_providers_mock.get.assert_called_with(service_fakes.sp_id, id='BETA')
    collist = ('auth_url', 'description', 'enabled', 'id', 'sp_url')
    self.assertEqual(collist, columns)
    datalist = (service_fakes.sp_auth_url, service_fakes.sp_description, True, service_fakes.sp_id, service_fakes.service_provider_url)
    self.assertEqual(data, datalist)