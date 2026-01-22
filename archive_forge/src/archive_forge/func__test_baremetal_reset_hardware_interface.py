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
def _test_baremetal_reset_hardware_interface(self, interface):
    arglist = ['node_uuid', '--reset-%s-interface' % interface]
    verifylist = [('nodes', ['node_uuid']), ('reset_%s_interface' % interface, True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/%s_interface' % interface, 'op': 'remove'}], reset_interfaces=None)