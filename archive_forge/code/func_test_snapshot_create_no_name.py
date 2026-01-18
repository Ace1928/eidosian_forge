from unittest import mock
import io
from osc_lib import exceptions as exc
from heatclient import exc as heat_exc
from heatclient.osc.v1 import snapshot
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
def test_snapshot_create_no_name(self):
    arglist = ['my_stack']
    parsed_args = self.check_parser(self.cmd, arglist, [])
    self.stack_client.snapshot.return_value = self.get_response
    self.cmd.take_action(parsed_args)
    self.stack_client.snapshot.assert_called_with('my_stack', None)