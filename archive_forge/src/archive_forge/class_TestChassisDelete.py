import copy
from unittest import mock
from osc_lib.tests import utils as oscutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_chassis
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
class TestChassisDelete(TestChassis):

    def setUp(self):
        super(TestChassisDelete, self).setUp()
        self.baremetal_mock.chassis.get.return_value = baremetal_fakes.FakeBaremetalResource(None, copy.deepcopy(baremetal_fakes.BAREMETAL_CHASSIS), loaded=True)
        self.cmd = baremetal_chassis.DeleteBaremetalChassis(self.app, None)

    def test_chassis_delete(self):
        uuid = baremetal_fakes.baremetal_chassis_uuid
        arglist = [uuid]
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.chassis.delete.assert_called_with(uuid)

    def test_chassis_delete_multiple(self):
        uuid1 = 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'
        uuid2 = '11111111-2222-3333-4444-555555555555'
        arglist = [uuid1, uuid2]
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        args = [uuid1, uuid2]
        self.baremetal_mock.chassis.delete.assert_has_calls([mock.call(x) for x in args])
        self.assertEqual(2, self.baremetal_mock.chassis.delete.call_count)

    def test_chassis_delete_multiple_with_failure(self):
        uuid1 = 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'
        uuid2 = '11111111-2222-3333-4444-555555555555'
        arglist = [uuid1, uuid2]
        verifylist = []
        self.baremetal_mock.chassis.delete.side_effect = ['', exc.ClientException]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exc.ClientException, self.cmd.take_action, parsed_args)
        args = [uuid1, uuid2]
        self.baremetal_mock.chassis.delete.assert_has_calls([mock.call(x) for x in args])
        self.assertEqual(2, self.baremetal_mock.chassis.delete.call_count)