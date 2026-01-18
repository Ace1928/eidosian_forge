import testtools
from osc_lib import exceptions
from osc_lib.tests import utils
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
def test_set_shared_and_no_shared(self):
    target = self.resource['id']
    arglist = [target, '--share', '--no-share']
    verifylist = [(self.res, target), ('share', True), ('no_share', True)]
    self.assertRaises(utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)