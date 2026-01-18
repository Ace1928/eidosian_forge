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
def test_show_project_image(self):
    arglist = [self._image.id, 'member1']
    verifylist = [('image', self._image.id), ('member', 'member1')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.image_client.find_image.assert_called_with(self._image.id, ignore_missing=False)
    self.image_client.get_member.assert_called_with(member='member1', image=self._image.id)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.datalist, data)