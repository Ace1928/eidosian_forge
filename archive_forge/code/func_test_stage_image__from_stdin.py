import copy
import io
import tempfile
from unittest import mock
from cinderclient import api_versions
from openstack import exceptions as sdk_exceptions
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.image.v2 import image as _image
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
@mock.patch('openstackclient.image.v2.image.get_data_from_stdin')
def test_stage_image__from_stdin(self, mock_get_data_from_stdin):
    fake_stdin = io.BytesIO(b'some initial binary data: \x00\x01')
    mock_get_data_from_stdin.return_value = fake_stdin
    arglist = [self.image.name]
    verifylist = [('image', self.image.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.image_client.stage_image.assert_called_once_with(self.image, data=fake_stdin)