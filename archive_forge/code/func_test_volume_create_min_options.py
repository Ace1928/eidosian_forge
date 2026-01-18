from unittest import mock
from unittest.mock import call
from cinderclient import api_versions
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume
def test_volume_create_min_options(self):
    arglist = ['--size', str(self.new_volume.size)]
    verifylist = [('size', self.new_volume.size)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.volumes_mock.create.assert_called_with(size=self.new_volume.size, snapshot_id=None, name=None, description=None, volume_type=None, availability_zone=None, metadata=None, imageRef=None, source_volid=None, consistencygroup_id=None, scheduler_hints=None, backup_id=None)
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.datalist, data)