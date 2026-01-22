import copy
import io
import json
import sys
from unittest import mock
from osc_lib.tests import utils as oscutils
from ironicclient.common import utils as commonutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_node
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
from ironicclient.v1 import utils as v1_utils
class TestBaremetalReboot(TestBaremetal):

    def setUp(self):
        super(TestBaremetalReboot, self).setUp()
        self.cmd = baremetal_node.RebootBaremetalNode(self.app, None)

    def test_baremetal_reboot_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(oscutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_baremetal_reboot_uuid_only(self):
        arglist = ['node_uuid']
        verifylist = [('nodes', ['node_uuid']), ('soft', False), ('power_timeout', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.set_power_state.assert_called_once_with('node_uuid', 'reboot', False, timeout=None)

    def test_baremetal_reboot_timeout(self):
        arglist = ['node_uuid', '--power-timeout', '2']
        verifylist = [('nodes', ['node_uuid']), ('soft', False), ('power_timeout', 2)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.set_power_state.assert_called_once_with('node_uuid', 'reboot', False, timeout=2)

    def test_baremetal_soft_reboot(self):
        arglist = ['node_uuid', '--soft']
        verifylist = [('nodes', ['node_uuid']), ('soft', True), ('power_timeout', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.set_power_state.assert_called_once_with('node_uuid', 'reboot', True, timeout=None)

    def test_baremetal_soft_reboot_timeout(self):
        arglist = ['node_uuid', '--soft', '--power-timeout', '2']
        verifylist = [('nodes', ['node_uuid']), ('soft', True), ('power_timeout', 2)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.set_power_state.assert_called_once_with('node_uuid', 'reboot', True, timeout=2)