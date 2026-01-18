import copy
from openstackclient.identity.v3 import region
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_region_create_parent_region_id(self):
    arglist = [identity_fakes.region_id, '--parent-region', identity_fakes.region_parent_region_id]
    verifylist = [('region', identity_fakes.region_id), ('parent_region', identity_fakes.region_parent_region_id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'description': None, 'id': identity_fakes.region_id, 'parent_region': identity_fakes.region_parent_region_id}
    self.regions_mock.create.assert_called_with(**kwargs)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.datalist, data)