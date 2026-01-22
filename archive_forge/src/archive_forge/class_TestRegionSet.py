import copy
from openstackclient.identity.v3 import region
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestRegionSet(TestRegion):

    def setUp(self):
        super(TestRegionSet, self).setUp()
        self.regions_mock.update.return_value = fakes.FakeResource(None, copy.deepcopy(identity_fakes.REGION), loaded=True)
        self.cmd = region.SetRegion(self.app, None)

    def test_region_set_no_options(self):
        arglist = [identity_fakes.region_id]
        verifylist = [('region', identity_fakes.region_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {}
        self.regions_mock.update.assert_called_with(identity_fakes.region_id, **kwargs)
        self.assertIsNone(result)

    def test_region_set_description(self):
        arglist = ['--description', 'qwerty', identity_fakes.region_id]
        verifylist = [('description', 'qwerty'), ('region', identity_fakes.region_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'description': 'qwerty'}
        self.regions_mock.update.assert_called_with(identity_fakes.region_id, **kwargs)
        self.assertIsNone(result)

    def test_region_set_parent_region_id(self):
        arglist = ['--parent-region', 'new_parent', identity_fakes.region_id]
        verifylist = [('parent_region', 'new_parent'), ('region', identity_fakes.region_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'parent_region': 'new_parent'}
        self.regions_mock.update.assert_called_with(identity_fakes.region_id, **kwargs)
        self.assertIsNone(result)