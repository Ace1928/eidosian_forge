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
class TestListImageProjects(TestImage):
    project = identity_fakes.FakeProject.create_one_project()
    _image = image_fakes.create_one_image()
    member = image_fakes.create_one_image_member(attrs={'image_id': _image.id, 'member_id': project.id})
    columns = ('Image ID', 'Member ID', 'Status')
    datalist = [(_image.id, member.member_id, member.status)]

    def setUp(self):
        super().setUp()
        self.image_client.find_image.return_value = self._image
        self.image_client.members.return_value = [self.member]
        self.cmd = _image.ListImageProjects(self.app, None)

    def test_image_member_list(self):
        arglist = [self._image.id]
        verifylist = [('image', self._image.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.image_client.members.assert_called_with(image=self._image.id)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, list(data))