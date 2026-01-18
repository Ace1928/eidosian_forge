import copy
from openstackclient.identity.v3 import region
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_region_set_no_options(self):
    arglist = [identity_fakes.region_id]
    verifylist = [('region', identity_fakes.region_id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    kwargs = {}
    self.regions_mock.update.assert_called_with(identity_fakes.region_id, **kwargs)
    self.assertIsNone(result)