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
class TestVifList(TestBaremetal):

    def setUp(self):
        super(TestVifList, self).setUp()
        self.baremetal_mock.node.vif_list.return_value = [baremetal_fakes.FakeBaremetalResource(None, copy.deepcopy(baremetal_fakes.VIFS), loaded=True)]
        self.cmd = baremetal_node.VifListBaremetalNode(self.app, None)

    def test_baremetal_vif_list(self):
        arglist = ['node_uuid']
        verifylist = [('node', 'node_uuid')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.vif_list.assert_called_once_with('node_uuid')