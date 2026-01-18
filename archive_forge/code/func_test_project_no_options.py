from io import StringIO
from unittest import mock
from openstackclient.common import project_cleanup
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as test_utils
def test_project_no_options(self):
    arglist = []
    verifylist = []
    self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)