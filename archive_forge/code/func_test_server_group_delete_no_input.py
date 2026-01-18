from unittest import mock
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.compute.v2 import server_group
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_server_group_delete_no_input(self):
    arglist = []
    verifylist = None
    self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)