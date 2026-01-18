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
def test_server_add_floating_ip_no_external_gateway(self, success=False):
    _server = compute_fakes.create_one_server()
    self.servers_mock.get.return_value = _server
    _port = network_fakes.create_one_port()
    _floating_ip = network_fakes.FakeFloatingIP.create_one_floating_ip()
    self.network_client.find_ip = mock.Mock(return_value=_floating_ip)
    return_value = [_port]
    if success:
        return_value.append(_port)
    self.network_client.ports = mock.Mock(return_value=return_value)
    side_effect = [sdk_exceptions.NotFoundException()]
    if success:
        side_effect.append(None)
    self.network_client.update_ip = mock.Mock(side_effect=side_effect)
    arglist = [_server.id, _floating_ip['floating_ip_address']]
    verifylist = [('server', _server.id), ('ip_address', _floating_ip['floating_ip_address'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    if success:
        self.cmd.take_action(parsed_args)
    else:
        self.assertRaises(sdk_exceptions.NotFoundException, self.cmd.take_action, parsed_args)
    attrs = {'port_id': _port.id}
    self.network_client.find_ip.assert_called_once_with(_floating_ip['floating_ip_address'], ignore_missing=False)
    self.network_client.ports.assert_called_once_with(device_id=_server.id)
    if success:
        self.assertEqual(2, self.network_client.update_ip.call_count)
        calls = [mock.call(_floating_ip, **attrs)] * 2
        self.network_client.update_ip.assert_has_calls(calls)
    else:
        self.network_client.update_ip.assert_called_once_with(_floating_ip, **attrs)