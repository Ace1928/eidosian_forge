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
class TestServerRescue(TestServer):

    def setUp(self):
        super(TestServerRescue, self).setUp()
        self.image = image_fakes.create_one_image()
        self.image_client.get_image.return_value = self.image
        new_server = compute_fakes.create_one_server()
        attrs = {'id': new_server.id, 'image': {'id': self.image.id}, 'networks': {}, 'adminPass': 'passw0rd'}
        methods = {'rescue': new_server}
        self.server = compute_fakes.create_one_server(attrs=attrs, methods=methods)
        self.servers_mock.get.return_value = self.server
        self.cmd = server.RescueServer(self.app, None)

    def test_rescue_with_current_image(self):
        arglist = [self.server.id]
        verifylist = [('server', self.server.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.servers_mock.get.assert_called_with(self.server.id)
        self.server.rescue.assert_called_with(image=None, password=None)

    def test_rescue_with_new_image(self):
        new_image = image_fakes.create_one_image()
        self.image_client.find_image.return_value = new_image
        arglist = ['--image', new_image.id, self.server.id]
        verifylist = [('image', new_image.id), ('server', self.server.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.servers_mock.get.assert_called_with(self.server.id)
        self.image_client.find_image.assert_called_with(new_image.id, ignore_missing=False)
        self.server.rescue.assert_called_with(image=new_image, password=None)

    def test_rescue_with_current_image_and_password(self):
        password = 'password-xxx'
        arglist = ['--password', password, self.server.id]
        verifylist = [('password', password), ('server', self.server.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.servers_mock.get.assert_called_with(self.server.id)
        self.server.rescue.assert_called_with(image=None, password=password)