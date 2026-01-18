import testtools
from osc_lib import exceptions
from osc_lib.tests import utils
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
def test_set_shared(self):
    target = self.resource['id']
    arglist = [target, '--share']
    verifylist = [(self.res, target), ('share', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.mocked.assert_called_once_with(target, **{'shared': True})
    self.assertIsNone(result)