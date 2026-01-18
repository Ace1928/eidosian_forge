from unittest import mock
from openstackclient.compute.v2 import console
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import utils
def test_show_no_args(self):
    arglist = []
    verifylist = []
    self.assertRaises(utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)