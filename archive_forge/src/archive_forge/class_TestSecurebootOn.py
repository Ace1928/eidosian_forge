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
class TestSecurebootOn(TestBaremetal):

    def setUp(self):
        super(TestSecurebootOn, self).setUp()
        self.cmd = baremetal_node.SecurebootOnBaremetalNode(self.app, None)

    def test_console_enable(self):
        arglist = ['node_uuid']
        verifylist = [('nodes', ['node_uuid'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.set_secure_boot.assert_called_once_with('node_uuid', 'on')