import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_allocation
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
class TestBaremetalAllocationUnset(TestBaremetalAllocation):

    def setUp(self):
        super(TestBaremetalAllocationUnset, self).setUp()
        self.baremetal_mock.allocation.update.return_value = baremetal_fakes.FakeBaremetalResource(None, copy.deepcopy(baremetal_fakes.ALLOCATION), loaded=True)
        self.cmd = baremetal_allocation.UnsetBaremetalAllocation(self.app, None)

    def test_baremetal_allocation_unset_name(self):
        arglist = [baremetal_fakes.baremetal_uuid, '--name']
        verifylist = [('allocation', baremetal_fakes.baremetal_uuid), ('name', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.allocation.update.assert_called_once_with(baremetal_fakes.baremetal_uuid, [{'path': '/name', 'op': 'remove'}])

    def test_baremetal_allocation_unset_extra(self):
        arglist = [baremetal_fakes.baremetal_uuid, '--extra', 'key1']
        verifylist = [('allocation', baremetal_fakes.baremetal_uuid), ('extra', ['key1'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.allocation.update.assert_called_once_with(baremetal_fakes.baremetal_uuid, [{'path': '/extra/key1', 'op': 'remove'}])

    def test_baremetal_allocation_unset_multiple_extras(self):
        arglist = [baremetal_fakes.baremetal_uuid, '--extra', 'key1', '--extra', 'key2']
        verifylist = [('allocation', baremetal_fakes.baremetal_uuid), ('extra', ['key1', 'key2'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.allocation.update.assert_called_once_with(baremetal_fakes.baremetal_uuid, [{'path': '/extra/key1', 'op': 'remove'}, {'path': '/extra/key2', 'op': 'remove'}])

    def test_baremetal_allocation_unset_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(osctestutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_baremetal_allocation_unset_no_property(self):
        uuid = baremetal_fakes.baremetal_uuid
        arglist = [uuid]
        verifylist = [('allocation', uuid)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.assertFalse(self.baremetal_mock.allocation.update.called)