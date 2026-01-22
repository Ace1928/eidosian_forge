import collections
import copy
import getpass
import json
import tempfile
from unittest import mock
from unittest.mock import call
import iso8601
from novaclient import api_versions
from openstack import exceptions as sdk_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils as common_utils
from openstackclient.compute.v2 import server
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
class TestServerRebuildVolumeBacked(TestServer):

    def setUp(self):
        super().setUp()
        self.new_image = image_fakes.create_one_image()
        self.image_client.find_image.return_value = self.new_image
        attrs = {'image': '', 'networks': {}, 'adminPass': 'passw0rd'}
        new_server = compute_fakes.create_one_server(attrs=attrs)
        attrs['id'] = new_server.id
        attrs['status'] = 'ACTIVE'
        methods = {'rebuild': new_server}
        self.server = compute_fakes.create_one_server(attrs=attrs, methods=methods)
        self.servers_mock.get.return_value = self.server
        self.cmd = server.RebuildServer(self.app, None)

    def test_rebuild_with_reimage_boot_volume(self):
        self.compute_client.api_version = api_versions.APIVersion('2.93')
        arglist = [self.server.id, '--reimage-boot-volume', '--image', self.new_image.id]
        verifylist = [('server', self.server.id), ('reimage_boot_volume', True), ('image', self.new_image.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.servers_mock.get.assert_called_with(self.server.id)
        self.server.rebuild.assert_called_with(self.new_image, None)

    def test_rebuild_with_no_reimage_boot_volume(self):
        self.compute_client.api_version = api_versions.APIVersion('2.93')
        arglist = [self.server.id, '--no-reimage-boot-volume', '--image', self.new_image.id]
        verifylist = [('server', self.server.id), ('reimage_boot_volume', False), ('image', self.new_image.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--reimage-boot-volume is required', str(exc))

    def test_rebuild_with_reimage_boot_volume_pre_v293(self):
        self.compute_client.api_version = api_versions.APIVersion('2.92')
        arglist = [self.server.id, '--reimage-boot-volume', '--image', self.new_image.id]
        verifylist = [('server', self.server.id), ('reimage_boot_volume', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-compute-api-version 2.93 or greater is required', str(exc))