import copy
from unittest import mock
import testscenarios
from heatclient import exc
from heatclient.osc.v1 import event
from heatclient.tests.unit.osc.v1 import fakes
from heatclient.v1 import events
def _test_not_found(self, error):
    arglist = ['my_stack', 'my_resource', '1234']
    parsed_args = self.check_parser(self.cmd, arglist, [])
    ex = self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)
    self.assertIn(error, str(ex))