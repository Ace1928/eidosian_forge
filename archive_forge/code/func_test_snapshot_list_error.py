from unittest import mock
import io
from osc_lib import exceptions as exc
from heatclient import exc as heat_exc
from heatclient.osc.v1 import snapshot
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
def test_snapshot_list_error(self):
    self.stack_client.snapshot_list.side_effect = heat_exc.HTTPNotFound()
    arglist = ['my_stack']
    parsed_args = self.check_parser(self.cmd, arglist, [])
    error = self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)
    self.assertEqual('Stack not found: my_stack', str(error))