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
def test_server_create_with_not_exist_security_group(self):
    arglist = ['--image', 'image1', '--flavor', 'flavor1', '--key-name', 'keyname', '--security-group', 'securitygroup', '--security-group', 'not_exist_sg', self.new_server.name]
    verifylist = [('image', 'image1'), ('flavor', 'flavor1'), ('key_name', 'keyname'), ('security_group', ['securitygroup', 'not_exist_sg']), ('server_name', self.new_server.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    fake_sg = network_fakes.FakeSecurityGroup.create_security_groups(count=1)
    fake_sg.append(exceptions.NotFound(code=404))
    mock_find_sg = network_fakes.FakeSecurityGroup.get_security_groups(fake_sg)
    self.app.client_manager.network.find_security_group = mock_find_sg
    self.assertRaises(exceptions.NotFound, self.cmd.take_action, parsed_args)
    mock_find_sg.assert_called_with('not_exist_sg', ignore_missing=False)