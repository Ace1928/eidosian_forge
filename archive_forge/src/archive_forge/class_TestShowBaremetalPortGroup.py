import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient.osc.v1 import baremetal_portgroup
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
class TestShowBaremetalPortGroup(TestBaremetalPortGroup):

    def setUp(self):
        super(TestShowBaremetalPortGroup, self).setUp()
        self.baremetal_mock.portgroup.get.return_value = baremetal_fakes.FakeBaremetalResource(None, copy.deepcopy(baremetal_fakes.PORTGROUP), loaded=True)
        self.baremetal_mock.portgroup.get_by_address.return_value = baremetal_fakes.FakeBaremetalResource(None, copy.deepcopy(baremetal_fakes.PORTGROUP), loaded=True)
        self.cmd = baremetal_portgroup.ShowBaremetalPortGroup(self.app, None)

    def test_baremetal_portgroup_show(self):
        arglist = ['ppp-gggggg-pppp']
        verifylist = [('portgroup', baremetal_fakes.baremetal_portgroup_uuid)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        args = ['ppp-gggggg-pppp']
        self.baremetal_mock.portgroup.get.assert_called_with(*args, fields=None)
        collist = ('address', 'extra', 'mode', 'name', 'node_uuid', 'properties', 'uuid')
        self.assertEqual(collist, columns)
        datalist = (baremetal_fakes.baremetal_portgroup_address, baremetal_fakes.baremetal_portgroup_extra, baremetal_fakes.baremetal_portgroup_mode, baremetal_fakes.baremetal_portgroup_name, baremetal_fakes.baremetal_uuid, baremetal_fakes.baremetal_portgroup_properties, baremetal_fakes.baremetal_portgroup_uuid)
        self.assertEqual(datalist, tuple(data))

    def test_baremetal_portgroup_show_address(self):
        arglist = ['--address', baremetal_fakes.baremetal_portgroup_address]
        verifylist = [('address', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        args = {baremetal_fakes.baremetal_portgroup_address}
        self.baremetal_mock.portgroup.get_by_address.assert_called_with(*args, fields=None)

    def test_baremetal_portgroup_show_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(osctestutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)