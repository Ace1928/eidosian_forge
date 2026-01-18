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
def test_rebuild_baremetal_provision_state_active_and_configdrive(self):
    arglist = ['node_uuid', '--config-drive', 'path/to/drive', '--deploy-steps', '[{"interface":"deploy"}]']
    verifylist = [('nodes', ['node_uuid']), ('provision_state', 'rebuild'), ('config_drive', 'path/to/drive'), ('deploy_steps', '[{"interface":"deploy"}]')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.baremetal_mock.node.set_provision_state.assert_called_once_with('node_uuid', 'rebuild', cleansteps=None, deploysteps=[{'interface': 'deploy'}], configdrive='path/to/drive', rescue_password=None)