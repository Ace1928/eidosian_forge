from unittest import mock
from osc_lib.cli import format_columns
from openstackclient.network.v2 import ip_availability
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_show_no_option(self):
    arglist = []
    verifylist = []
    self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)