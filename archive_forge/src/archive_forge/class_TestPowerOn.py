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
class TestPowerOn(TestBaremetal):

    def setUp(self):
        super(TestPowerOn, self).setUp()
        self.cmd = baremetal_node.PowerOnBaremetalNode(self.app, None)

    def test_baremetal_power_on(self):
        arglist = ['node_uuid']
        verifylist = [('nodes', ['node_uuid']), ('power_timeout', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.set_power_state.assert_called_once_with('node_uuid', 'on', False, timeout=None)

    def test_baremetal_power_on_timeout(self):
        arglist = ['node_uuid', '--power-timeout', '2']
        verifylist = [('nodes', ['node_uuid']), ('power_timeout', 2)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.set_power_state.assert_called_once_with('node_uuid', 'on', False, timeout=2)

    def test_baremetal_power_on_no_args(self):
        arglist = []
        verifylist = []
        self.assertRaises(oscutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)