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
class TestRemoveProjectImage(TestImage):
    project = identity_fakes.FakeProject.create_one_project()
    domain = identity_fakes.FakeDomain.create_one_domain()

    def setUp(self):
        super().setUp()
        self._image = image_fakes.create_one_image()
        self.image_client.find_image.return_value = self._image
        self.project_mock.get.return_value = self.project
        self.domain_mock.get.return_value = self.domain
        self.image_client.remove_member.return_value = None
        self.cmd = _image.RemoveProjectImage(self.app, None)

    def test_remove_project_image_no_options(self):
        arglist = [self._image.id, self.project.id]
        verifylist = [('image', self._image.id), ('project', self.project.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.image_client.find_image.assert_called_with(self._image.id, ignore_missing=False)
        self.image_client.remove_member.assert_called_with(member=self.project.id, image=self._image.id)
        self.assertIsNone(result)

    def test_remove_project_image_with_options(self):
        arglist = [self._image.id, self.project.id, '--project-domain', self.domain.id]
        verifylist = [('image', self._image.id), ('project', self.project.id), ('project_domain', self.domain.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.image_client.remove_member.assert_called_with(member=self.project.id, image=self._image.id)
        self.assertIsNone(result)