import copy
from openstackclient.identity.v3 import region
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_region_show(self):
    arglist = [identity_fakes.region_id]
    verifylist = [('region', identity_fakes.region_id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.regions_mock.get.assert_called_with(identity_fakes.region_id)
    collist = ('description', 'parent_region', 'region')
    self.assertEqual(collist, columns)
    datalist = (identity_fakes.region_description, identity_fakes.region_parent_region_id, identity_fakes.region_id)
    self.assertEqual(datalist, data)