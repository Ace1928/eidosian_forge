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
class TestRebuildBaremetalProvisionState(TestBaremetal):

    def setUp(self):
        super(TestRebuildBaremetalProvisionState, self).setUp()
        self.cmd = baremetal_node.RebuildBaremetalNode(self.app, None)

    def test_rebuild_baremetal_provision_state_active_and_configdrive(self):
        arglist = ['node_uuid', '--config-drive', 'path/to/drive', '--deploy-steps', '[{"interface":"deploy"}]']
        verifylist = [('nodes', ['node_uuid']), ('provision_state', 'rebuild'), ('config_drive', 'path/to/drive'), ('deploy_steps', '[{"interface":"deploy"}]')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.set_provision_state.assert_called_once_with('node_uuid', 'rebuild', cleansteps=None, deploysteps=[{'interface': 'deploy'}], configdrive='path/to/drive', rescue_password=None)

    def test_rebuild_no_wait(self):
        arglist = ['node_uuid']
        verifylist = [('nodes', ['node_uuid']), ('provision_state', 'rebuild')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.set_provision_state.assert_called_once_with('node_uuid', 'rebuild', cleansteps=None, deploysteps=None, configdrive=None, rescue_password=None)
        self.baremetal_mock.node.wait_for_provision_state.assert_not_called()

    def test_rebuild_baremetal_provision_state_active_and_wait(self):
        arglist = ['node_uuid', '--wait', '15']
        verifylist = [('nodes', ['node_uuid']), ('provision_state', 'rebuild'), ('wait_timeout', 15)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        test_node = self.baremetal_mock.node
        test_node.wait_for_provision_state.assert_called_once_with(['node_uuid'], expected_state='active', poll_interval=10, timeout=15)

    def test_rebuild_baremetal_provision_state_default_wait(self):
        arglist = ['node_uuid', '--wait']
        verifylist = [('nodes', ['node_uuid']), ('provision_state', 'rebuild'), ('wait_timeout', 0)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        test_node = self.baremetal_mock.node
        test_node.wait_for_provision_state.assert_called_once_with(['node_uuid'], expected_state='active', poll_interval=10, timeout=0)