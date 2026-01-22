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
def _test_baremetal_list_maintenance(self, maint_option, maint_value):
    arglist = [maint_option]
    verifylist = [('maintenance', maint_value)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    kwargs = {'marker': None, 'limit': None, 'maintenance': maint_value}
    self.baremetal_mock.node.list.assert_called_with(**kwargs)