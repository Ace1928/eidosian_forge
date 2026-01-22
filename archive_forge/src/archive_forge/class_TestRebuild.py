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
class TestRebuild(TestBaremetal):

    def setUp(self):
        super(TestRebuild, self).setUp()
        self.cmd = baremetal_node.RebuildBaremetalNode(self.app, None)

    def test_rebuild(self):
        arglist = ['node_uuid']
        verifylist = [('nodes', ['node_uuid']), ('provision_state', 'rebuild')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.set_provision_state.assert_called_once_with('node_uuid', 'rebuild', cleansteps=None, configdrive=None, deploysteps=None, rescue_password=None)