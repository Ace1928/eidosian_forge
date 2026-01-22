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
class TestServerGeneral(TestServer):
    OLD = {'private': [{'addr': '192.168.0.3', 'version': 4}]}
    NEW = {'foo': [{'OS-EXT-IPS-MAC:mac_addr': 'fa:16:3e:93:b3:01', 'version': 4, 'addr': '10.10.1.2', 'OS-EXT-IPS:type': 'fixed'}, {'OS-EXT-IPS-MAC:mac_addr': 'fa:16:3e:93:b3:02', 'version': 6, 'addr': '0:0:0:0:0:ffff:a0a:103', 'OS-EXT-IPS:type': 'floating'}]}
    ODD = {'jenkins': ['10.3.3.18', '124.12.125.4']}

    def test_get_ip_address(self):
        self.assertEqual('192.168.0.3', server._get_ip_address(self.OLD, 'private', [4, 6]))
        self.assertEqual('10.10.1.2', server._get_ip_address(self.NEW, 'fixed', [4, 6]))
        self.assertEqual('10.10.1.2', server._get_ip_address(self.NEW, 'private', [4, 6]))
        self.assertEqual('0:0:0:0:0:ffff:a0a:103', server._get_ip_address(self.NEW, 'public', [6]))
        self.assertEqual('0:0:0:0:0:ffff:a0a:103', server._get_ip_address(self.NEW, 'floating', [6]))
        self.assertEqual('124.12.125.4', server._get_ip_address(self.ODD, 'public', [4, 6]))
        self.assertEqual('10.3.3.18', server._get_ip_address(self.ODD, 'private', [4, 6]))
        self.assertRaises(exceptions.CommandError, server._get_ip_address, self.NEW, 'public', [4])
        self.assertRaises(exceptions.CommandError, server._get_ip_address, self.NEW, 'admin', [4])
        self.assertRaises(exceptions.CommandError, server._get_ip_address, self.OLD, 'public', [4, 6])
        self.assertRaises(exceptions.CommandError, server._get_ip_address, self.OLD, 'private', [6])

    @mock.patch('osc_lib.utils.find_resource')
    def test_prep_server_detail(self, find_resource):
        _image = image_fakes.create_one_image()
        _flavor = compute_fakes.create_one_flavor()
        server_info = {'image': {u'id': _image.id}, 'flavor': {u'id': _flavor.id}, 'tenant_id': u'tenant-id-xxx', 'addresses': {u'public': [u'10.20.30.40', u'2001:db8::f']}, 'links': u'http://xxx.yyy.com', 'properties': '', 'volumes_attached': [{'id': '6344fe9d-ef20-45b2-91a6'}]}
        _server = compute_fakes.create_one_server(attrs=server_info)
        find_resource.side_effect = [_server, _flavor]
        self.image_client.get_image.return_value = _image
        info = {'id': _server.id, 'name': _server.name, 'image': '%s (%s)' % (_image.name, _image.id), 'flavor': '%s (%s)' % (_flavor.name, _flavor.id), 'OS-EXT-STS:power_state': server.PowerStateColumn(getattr(_server, 'OS-EXT-STS:power_state')), 'properties': '', 'volumes_attached': [{'id': '6344fe9d-ef20-45b2-91a6'}], 'addresses': format_columns.DictListColumn(_server.addresses), 'project_id': 'tenant-id-xxx'}
        server_detail = server._prep_server_detail(self.compute_client, self.image_client, _server)
        self.assertCountEqual(info, server_detail)