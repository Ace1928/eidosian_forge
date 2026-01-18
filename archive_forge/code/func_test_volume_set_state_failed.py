from unittest import mock
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.identity.v3 import fakes as project_fakes
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_snapshot
def test_volume_set_state_failed(self):
    self.snapshots_mock.reset_state.side_effect = exceptions.CommandError()
    arglist = ['--state', 'error', self.snapshot.id]
    verifylist = [('state', 'error'), ('snapshot', self.snapshot.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    try:
        self.cmd.take_action(parsed_args)
        self.fail('CommandError should be raised.')
    except exceptions.CommandError as e:
        self.assertEqual('One or more of the set operations failed', str(e))
    self.snapshots_mock.reset_state.assert_called_once_with(self.snapshot.id, 'error')