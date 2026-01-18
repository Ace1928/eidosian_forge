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
def test_volume_create_hints(self):
    """--hint needs to behave differently based on the given hint

        different_host and same_host need to append to a list if given multiple
        times. All other parameter are strings.
        """
    arglist = ['--size', str(self.new_volume.size), '--hint', 'k=v', '--hint', 'k=v2', '--hint', 'same_host=v3', '--hint', 'same_host=v4', '--hint', 'different_host=v5', '--hint', 'local_to_instance=v6', '--hint', 'different_host=v7', self.new_volume.name]
    verifylist = [('size', self.new_volume.size), ('hint', {'k': 'v2', 'same_host': ['v3', 'v4'], 'local_to_instance': 'v6', 'different_host': ['v5', 'v7']}), ('name', self.new_volume.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.volumes_mock.create.assert_called_with(size=self.new_volume.size, snapshot_id=None, name=self.new_volume.name, description=None, volume_type=None, availability_zone=None, metadata=None, imageRef=None, source_volid=None, consistencygroup_id=None, scheduler_hints={'k': 'v2', 'same_host': ['v3', 'v4'], 'local_to_instance': 'v6', 'different_host': ['v5', 'v7']}, backup_id=None)
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.datalist, data)