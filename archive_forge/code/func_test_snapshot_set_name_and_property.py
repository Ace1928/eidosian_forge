from unittest import mock
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.identity.v3 import fakes as project_fakes
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_snapshot
def test_snapshot_set_name_and_property(self):
    arglist = ['--name', 'new_snapshot', '--property', 'x=y', '--property', 'foo=foo', self.snapshot.id]
    new_property = {'x': 'y', 'foo': 'foo'}
    verifylist = [('name', 'new_snapshot'), ('property', new_property), ('snapshot', self.snapshot.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    kwargs = {'name': 'new_snapshot'}
    self.snapshots_mock.update.assert_called_with(self.snapshot.id, **kwargs)
    self.snapshots_mock.set_metadata.assert_called_with(self.snapshot.id, new_property)
    self.assertIsNone(result)