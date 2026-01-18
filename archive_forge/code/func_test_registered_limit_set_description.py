import copy
from keystoneauth1.exceptions import http as ksa_exceptions
from osc_lib import exceptions
from openstackclient.identity.v3 import registered_limit
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_registered_limit_set_description(self):
    registered_limit = copy.deepcopy(identity_fakes.REGISTERED_LIMIT)
    registered_limit['description'] = identity_fakes.registered_limit_description
    self.registered_limit_mock.update.return_value = fakes.FakeResource(None, registered_limit, loaded=True)
    arglist = ['--description', identity_fakes.registered_limit_description, identity_fakes.registered_limit_id]
    verifylist = [('description', identity_fakes.registered_limit_description), ('registered_limit_id', identity_fakes.registered_limit_id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.registered_limit_mock.update.assert_called_with(identity_fakes.registered_limit_id, service=None, resource_name=None, default_limit=None, description=identity_fakes.registered_limit_description, region=None)
    collist = ('default_limit', 'description', 'id', 'region_id', 'resource_name', 'service_id')
    self.assertEqual(collist, columns)
    datalist = (identity_fakes.registered_limit_default_limit, identity_fakes.registered_limit_description, identity_fakes.registered_limit_id, None, identity_fakes.registered_limit_resource_name, identity_fakes.service_id)
    self.assertEqual(datalist, data)