from unittest import mock
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.identity.v3 import fakes as project_fakes
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_snapshot
def test_snapshot_create_without_name(self):
    arglist = ['--volume', self.new_snapshot.volume_id]
    verifylist = [('volume', self.new_snapshot.volume_id)]
    self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)