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
def test_volume_create_options(self):
    consistency_group = volume_fakes.create_one_consistency_group()
    self.consistencygroups_mock.get.return_value = consistency_group
    arglist = ['--size', str(self.new_volume.size), '--description', self.new_volume.description, '--type', self.new_volume.volume_type, '--availability-zone', self.new_volume.availability_zone, '--consistency-group', consistency_group.id, '--hint', 'k=v', self.new_volume.name]
    verifylist = [('size', self.new_volume.size), ('description', self.new_volume.description), ('type', self.new_volume.volume_type), ('availability_zone', self.new_volume.availability_zone), ('consistency_group', consistency_group.id), ('hint', {'k': 'v'}), ('name', self.new_volume.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.volumes_mock.create.assert_called_with(size=self.new_volume.size, snapshot_id=None, name=self.new_volume.name, description=self.new_volume.description, volume_type=self.new_volume.volume_type, availability_zone=self.new_volume.availability_zone, metadata=None, imageRef=None, source_volid=None, consistencygroup_id=consistency_group.id, scheduler_hints={'k': 'v'}, backup_id=None)
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.datalist, data)