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
class TestServerLock(TestServer):

    def setUp(self):
        super().setUp()
        self.server = compute_fakes.create_one_sdk_server()
        self.compute_sdk_client.find_server.return_value = self.server
        self.compute_sdk_client.lock_server.return_value = None
        self.cmd = server.LockServer(self.app, None)

    @mock.patch.object(sdk_utils, 'supports_microversion')
    def test_server_lock(self, sm_mock):
        sm_mock.return_value = False
        self.run_method_with_sdk_servers('lock_server', 1)

    @mock.patch.object(sdk_utils, 'supports_microversion')
    def test_server_lock_multi_servers(self, sm_mock):
        sm_mock.return_value = False
        self.run_method_with_sdk_servers('lock_server', 3)

    @mock.patch.object(sdk_utils, 'supports_microversion')
    def test_server_lock_with_reason(self, sm_mock):
        sm_mock.return_value = True
        arglist = [self.server.id, '--reason', 'blah']
        verifylist = [('server', [self.server.id]), ('reason', 'blah')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.compute_sdk_client.find_server.assert_called_with(self.server.id, ignore_missing=False)
        self.compute_sdk_client.lock_server.assert_called_with(self.server.id, locked_reason='blah')

    @mock.patch.object(sdk_utils, 'supports_microversion')
    def test_server_lock_with_reason_multi_servers(self, sm_mock):
        sm_mock.return_value = True
        server2 = compute_fakes.create_one_sdk_server()
        arglist = [self.server.id, server2.id, '--reason', 'choo..choo']
        verifylist = [('server', [self.server.id, server2.id]), ('reason', 'choo..choo')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.assertEqual(2, self.compute_sdk_client.find_server.call_count)
        self.compute_sdk_client.lock_server.assert_called_with(self.server.id, locked_reason='choo..choo')
        self.assertEqual(2, self.compute_sdk_client.lock_server.call_count)

    @mock.patch.object(sdk_utils, 'supports_microversion')
    def test_server_lock_with_reason_pre_v273(self, sm_mock):
        sm_mock.return_value = False
        server = compute_fakes.create_one_sdk_server()
        arglist = [server.id, '--reason', 'blah']
        verifylist = [('server', [server.id]), ('reason', 'blah')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-compute-api-version 2.73 or greater is required', str(ex))