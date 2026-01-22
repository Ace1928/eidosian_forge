import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_allocation
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
class TestBaremetalAllocationSet(TestBaremetalAllocation):

    def setUp(self):
        super(TestBaremetalAllocationSet, self).setUp()
        self.baremetal_mock.allocation.update.return_value = baremetal_fakes.FakeBaremetalResource(None, copy.deepcopy(baremetal_fakes.ALLOCATION), loaded=True)
        self.cmd = baremetal_allocation.SetBaremetalAllocation(self.app, None)

    def test_baremetal_allocation_set_name(self):
        new_name = 'foo'
        arglist = [baremetal_fakes.baremetal_uuid, '--name', new_name]
        verifylist = [('allocation', baremetal_fakes.baremetal_uuid), ('name', new_name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.allocation.update.assert_called_once_with(baremetal_fakes.baremetal_uuid, [{'path': '/name', 'value': new_name, 'op': 'add'}])

    def test_baremetal_allocation_set_extra(self):
        extra_value = 'foo=bar'
        arglist = [baremetal_fakes.baremetal_uuid, '--extra', extra_value]
        verifylist = [('allocation', baremetal_fakes.baremetal_uuid), ('extra', [extra_value])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.allocation.update.assert_called_once_with(baremetal_fakes.baremetal_uuid, [{'path': '/extra/foo', 'value': 'bar', 'op': 'add'}])

    def test_baremetal_allocation_set_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(osctestutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)