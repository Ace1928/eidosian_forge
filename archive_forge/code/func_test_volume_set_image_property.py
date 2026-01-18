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
def test_volume_set_image_property(self):
    arglist = ['--image-property', 'Alpha=a', '--image-property', 'Beta=b', self.new_volume.id]
    verifylist = [('image_property', {'Alpha': 'a', 'Beta': 'b'}), ('volume', self.new_volume.id), ('bootable', False), ('non_bootable', False)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.volumes_mock.set_image_metadata.assert_called_with(self.new_volume.id, parsed_args.image_property)