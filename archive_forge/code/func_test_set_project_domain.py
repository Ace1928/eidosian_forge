import testtools
from osc_lib import exceptions
from osc_lib.tests import utils
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
def test_set_project_domain(self):
    target = self.resource['id']
    project_domain = 'mydomain.com'
    arglist = [target, '--project-domain', project_domain]
    verifylist = [(self.res, target), ('project_domain', project_domain)]
    self.assertRaises(utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)