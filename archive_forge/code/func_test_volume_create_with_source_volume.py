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
def test_volume_create_with_source_volume(self):
    source_vol = 'source_vol'
    arglist = ['--source', self.new_volume.id, source_vol]
    verifylist = [('source', self.new_volume.id), ('name', source_vol)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.volumes_mock.get.return_value = self.new_volume
    columns, data = self.cmd.take_action(parsed_args)
    self.volumes_mock.create.assert_called_once_with(size=self.new_volume.size, snapshot_id=None, name=source_vol, description=None, volume_type=None, availability_zone=None, metadata=None, imageRef=None, source_volid=self.new_volume.id, consistencygroup_id=None, scheduler_hints=None, backup_id=None)
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.datalist, data)