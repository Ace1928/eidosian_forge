import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient.osc.v1 import baremetal_portgroup
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
class TestBaremetalPortGroupDelete(TestBaremetalPortGroup):

    def setUp(self):
        super(TestBaremetalPortGroupDelete, self).setUp()
        self.baremetal_mock.portgroup.get.return_value = baremetal_fakes.FakeBaremetalResource(None, copy.deepcopy(baremetal_fakes.PORTGROUP), loaded=True)
        self.cmd = baremetal_portgroup.DeleteBaremetalPortGroup(self.app, None)

    def test_baremetal_portgroup_delete(self):
        arglist = [baremetal_fakes.baremetal_portgroup_uuid]
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        args = baremetal_fakes.baremetal_portgroup_uuid
        self.baremetal_mock.portgroup.delete.assert_called_with(args)

    def test_baremetal_portgroup_delete_multiple(self):
        arglist = [baremetal_fakes.baremetal_portgroup_uuid, baremetal_fakes.baremetal_portgroup_name]
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        args = [baremetal_fakes.baremetal_portgroup_uuid, baremetal_fakes.baremetal_portgroup_name]
        self.baremetal_mock.portgroup.delete.assert_has_calls([mock.call(x) for x in args])
        self.assertEqual(2, self.baremetal_mock.portgroup.delete.call_count)

    def test_baremetal_portgroup_delete_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(osctestutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)