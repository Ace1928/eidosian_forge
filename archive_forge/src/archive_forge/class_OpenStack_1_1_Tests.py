import os
import sys
import datetime
import unittest
from unittest import mock
from unittest.mock import Mock, patch
import pytest
import requests_mock
from libcloud.test import XML_HEADERS, MockHttp
from libcloud.pricing import set_pricing, clear_pricing_data
from libcloud.utils.py3 import u, httplib, method_type
from libcloud.common.base import LibcloudConnection
from libcloud.common.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.compute.base import Node, NodeSize, NodeImage
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import OPENSTACK_PARAMS
from libcloud.compute.types import (
from libcloud.utils.iso8601 import UTC
from libcloud.common.exceptions import BaseHTTPError
from libcloud.compute.providers import get_driver
from libcloud.test.file_fixtures import OpenStackFixtures, ComputeFileFixtures
from libcloud.common.openstack_identity import (
from libcloud.compute.drivers.openstack import (
class OpenStack_1_1_Tests(unittest.TestCase, TestCaseMixin):
    should_list_locations = False
    should_list_volumes = True
    driver_klass = OpenStack_1_1_NodeDriver
    driver_type = OpenStack_1_1_NodeDriver
    driver_args = OPENSTACK_PARAMS
    driver_kwargs = {'ex_force_auth_version': '2.0'}

    @classmethod
    def create_driver(self):
        if self is not OpenStack_1_1_FactoryMethodTests:
            self.driver_type = self.driver_klass
        return self.driver_type(*self.driver_args, **self.driver_kwargs)

    def setUp(self):
        self.driver_klass.connectionCls.conn_class = OpenStack_2_0_MockHttp
        self.driver_klass.connectionCls.auth_url = 'https://auth.api.example.com'
        OpenStackMockHttp.type = None
        OpenStack_1_1_MockHttp.type = None
        OpenStack_2_0_MockHttp.type = None
        self.driver = self.create_driver()
        self.driver.connection._populate_hosts_and_request_paths()
        clear_pricing_data()
        self.node = self.driver.list_nodes()[1]

    def _force_reauthentication(self):
        """
        Trash current auth token so driver will be forced to re-authenticate
        on next request.
        """
        self.driver.connection._ex_force_base_url = 'http://ex_force_base_url.com:666/forced_url'
        self.driver.connection.auth_token = None
        self.driver.connection.auth_token_expires = None
        self.driver.connection._osa.auth_token = None
        self.driver.connection._osa.auth_token_expires = None

    def test_auth_token_is_set(self):
        self._force_reauthentication()
        self.driver.connection._populate_hosts_and_request_paths()
        self.assertEqual(self.driver.connection.auth_token, 'aaaaaaaaaaaa-bbb-cccccccccccccc')

    def test_auth_token_expires_is_set(self):
        self._force_reauthentication()
        self.driver.connection._populate_hosts_and_request_paths()
        expires = self.driver.connection.auth_token_expires
        self.assertEqual(expires.isoformat(), '2999-11-23T21:00:14-06:00')

    def test_ex_force_base_url(self):
        self.driver.connection._ex_force_base_url = 'http://ex_force_base_url.com:666/forced_url'
        self.driver.connection.auth_token = None
        self.driver.connection._populate_hosts_and_request_paths()
        self.assertEqual(self.driver.connection.host, 'ex_force_base_url.com')
        self.assertEqual(self.driver.connection.port, 666)
        self.assertEqual(self.driver.connection.request_path, '/forced_url')

    def test_get_endpoint_populates_host_port_and_request_path(self):
        self.driver.connection.get_endpoint = lambda: 'http://endpoint_auth_url.com:1555/service_url'
        self.driver.connection.auth_token = None
        self.driver.connection._ex_force_base_url = None
        self.driver.connection._populate_hosts_and_request_paths()
        self.assertEqual(self.driver.connection.host, 'endpoint_auth_url.com')
        self.assertEqual(self.driver.connection.port, 1555)
        self.assertEqual(self.driver.connection.request_path, '/service_url')

    def test_set_auth_token_populates_host_port_and_request_path(self):
        self.driver.connection._ex_force_base_url = 'http://some_other_ex_force_base_url.com:1222/some-service'
        self.driver.connection.auth_token = 'preset-auth-token'
        self.driver.connection._populate_hosts_and_request_paths()
        self.assertEqual(self.driver.connection.host, 'some_other_ex_force_base_url.com')
        self.assertEqual(self.driver.connection.port, 1222)
        self.assertEqual(self.driver.connection.request_path, '/some-service')

    def test_auth_token_without_base_url_raises_exception(self):
        kwargs = {'ex_force_auth_version': '2.0', 'ex_force_auth_token': 'preset-auth-token'}
        try:
            self.driver_type(*self.driver_args, **kwargs)
            self.fail('Expected failure setting auth token without base url')
        except LibcloudError:
            pass
        else:
            self.fail('Expected failure setting auth token without base url')

    def test_ex_force_auth_token_passed_to_connection(self):
        base_url = 'https://servers.api.rackspacecloud.com/v1.1/slug'
        kwargs = {'ex_force_auth_version': '2.0', 'ex_force_auth_token': 'preset-auth-token', 'ex_force_base_url': base_url}
        driver = self.driver_type(*self.driver_args, **kwargs)
        driver.list_nodes()
        self.assertEqual(kwargs['ex_force_auth_token'], driver.connection.auth_token)
        self.assertEqual('servers.api.rackspacecloud.com', driver.connection.host)
        self.assertEqual('/v1.1/slug', driver.connection.request_path)
        self.assertEqual(443, driver.connection.port)

    def test_ex_auth_cache_passed_to_identity_connection(self):
        kwargs = self.driver_kwargs.copy()
        kwargs['ex_auth_cache'] = OpenStackMockAuthCache()
        driver = self.driver_type(*self.driver_args, **kwargs)
        driver.connection.get_auth_class()
        driver.list_nodes()
        self.assertEqual(kwargs['ex_auth_cache'], driver.connection.get_auth_class().auth_cache)

    def test_unauthorized_clears_cached_auth_context(self):
        auth_cache = OpenStackMockAuthCache()
        self.assertEqual(len(auth_cache), 0)
        kwargs = self.driver_kwargs.copy()
        kwargs['ex_auth_cache'] = auth_cache
        driver = self.driver_type(*self.driver_args, **kwargs)
        driver.list_nodes()
        self.assertEqual(len(auth_cache), 1)
        self.driver_klass.connectionCls.conn_class.type = 'UNAUTHORIZED'
        with pytest.raises(BaseHTTPError):
            driver.list_nodes()
        self.assertEqual(len(auth_cache), 0)

    def test_list_nodes(self):
        nodes = self.driver.list_nodes()
        self.assertEqual(len(nodes), 2)
        node = nodes[0]
        self.assertEqual('12065', node.id)
        self.assertTrue('12.16.18.28' in node.public_ips)
        self.assertTrue('50.57.94.35' in node.public_ips)
        self.assertTrue('1.1.1.1' in node.public_ips)
        self.assertTrue('2.2.2.2' in node.public_ips)
        self.assertTrue('2001:4801:7808:52:16:3eff:fe47:788a' in node.public_ips)
        self.assertTrue('10.182.64.34' in node.private_ips)
        self.assertTrue('10.3.3.3' in node.private_ips)
        self.assertTrue('192.168.3.3' in node.private_ips)
        self.assertTrue('172.16.1.1' in node.private_ips)
        self.assertTrue('fec0:4801:7808:52:16:3eff:fe60:187d' in node.private_ips)
        self.assertEqual(node.created_at, datetime.datetime(2011, 10, 11, 0, 51, 39, tzinfo=UTC))
        self.assertEqual(node.extra.get('flavorId'), '2')
        self.assertEqual(node.extra.get('imageId'), '7')
        self.assertEqual(node.extra.get('metadata'), {})
        self.assertEqual(node.extra['updated'], '2011-10-11T00:50:04Z')
        self.assertEqual(node.extra['created'], '2011-10-11T00:51:39Z')
        self.assertEqual(node.extra.get('userId'), 'rs-reach')
        self.assertEqual(node.extra.get('hostId'), '912566d83a13fbb357ea3f13c629363d9f7e1ba3f925b49f3d2ab725')
        self.assertEqual(node.extra.get('disk_config'), 'AUTO')
        self.assertEqual(node.extra.get('task_state'), 'spawning')
        self.assertEqual(node.extra.get('vm_state'), 'active')
        self.assertEqual(node.extra.get('power_state'), 1)
        self.assertEqual(node.extra.get('progress'), 25)
        self.assertEqual(node.extra.get('fault')['id'], 1234)
        self.assertTrue(node.extra.get('service_name') is not None)
        self.assertTrue(node.extra.get('uri') is not None)

    def test_list_nodes_no_image_id_attribute(self):
        self.driver_klass.connectionCls.conn_class.type = 'ERROR_STATE_NO_IMAGE_ID'
        nodes = self.driver.list_nodes()
        self.assertIsNone(nodes[0].extra['imageId'])

    def test_list_volumes(self):
        volumes = self.driver.list_volumes()
        self.assertEqual(len(volumes), 2)
        volume = volumes[0]
        self.assertEqual('cd76a3a1-c4ce-40f6-9b9f-07a61508938d', volume.id)
        self.assertEqual('test_volume_2', volume.name)
        self.assertEqual(StorageVolumeState.AVAILABLE, volume.state)
        self.assertEqual(2, volume.size)
        self.assertEqual(volume.extra, {'description': '', 'attachments': [{'id': 'cd76a3a1-c4ce-40f6-9b9f-07a61508938d', 'device': '/dev/vdb', 'serverId': '12065', 'volumeId': 'cd76a3a1-c4ce-40f6-9b9f-07a61508938d'}], 'snapshot_id': None, 'state': 'available', 'location': 'nova', 'volume_type': 'None', 'metadata': {}, 'created_at': '2013-06-24T11:20:13.000000'})
        volume = volumes[1]
        self.assertEqual('cfcec3bc-b736-4db5-9535-4c24112691b5', volume.id)
        self.assertEqual('test_volume', volume.name)
        self.assertEqual(50, volume.size)
        self.assertEqual(StorageVolumeState.UNKNOWN, volume.state)
        self.assertEqual(volume.extra, {'description': 'some description', 'attachments': [], 'snapshot_id': '01f48111-7866-4cd2-986a-e92683c4a363', 'state': 'some-unknown-state', 'location': 'nova', 'volume_type': 'None', 'metadata': {}, 'created_at': '2013-06-21T12:39:02.000000'})

    def test_list_sizes(self):
        sizes = self.driver.list_sizes()
        self.assertEqual(len(sizes), 8, 'Wrong sizes count')
        for size in sizes:
            self.assertTrue(size.price is None or isinstance(size.price, float), 'Wrong size price type')
            self.assertTrue(isinstance(size.ram, int))
            self.assertTrue(isinstance(size.vcpus, int))
            self.assertTrue(isinstance(size.disk, int))
            self.assertTrue(isinstance(size.swap, int))
            self.assertTrue(isinstance(size.ephemeral_disk, int) or size.ephemeral_disk is None)
            self.assertTrue(isinstance(size.extra, dict))
            if size.id == '1':
                self.assertEqual(size.ephemeral_disk, 40)
                self.assertEqual(size.extra, {'policy_class': 'standard_flavor', 'class': 'standard1', 'disk_io_index': '2', 'number_of_data_disks': '0', 'disabled': False})
        self.assertEqual(sizes[0].vcpus, 8)

    def test_list_sizes_with_specified_pricing(self):
        pricing = {str(i): i * 5.0 for i in range(1, 9)}
        set_pricing(driver_type='compute', driver_name=self.driver.api_name, pricing=pricing)
        sizes = self.driver.list_sizes()
        self.assertEqual(len(sizes), 8, 'Wrong sizes count')
        for size in sizes:
            self.assertTrue(isinstance(size.price, float), 'Wrong size price type')
            self.assertEqual(size.price, pricing[size.id], 'Size price should match')

    def test_list_images(self):
        images = self.driver.list_images()
        self.assertEqual(len(images), 13, 'Wrong images count')
        image = images[0]
        self.assertEqual(image.id, '13')
        self.assertEqual(image.name, 'Windows 2008 SP2 x86 (B24)')
        self.assertEqual(image.extra['updated'], '2011-08-06T18:14:02Z')
        self.assertEqual(image.extra['created'], '2011-08-06T18:13:11Z')
        self.assertEqual(image.extra['status'], 'ACTIVE')
        self.assertEqual(image.extra['metadata']['os_type'], 'windows')
        self.assertEqual(image.extra['serverId'], '52415800-8b69-11e0-9b19-734f335aa7b3')
        self.assertEqual(image.extra['minDisk'], 0)
        self.assertEqual(image.extra['minRam'], 0)

    def test_create_node(self):
        image = NodeImage(id=11, name='Ubuntu 8.10 (intrepid)', driver=self.driver)
        size = NodeSize(1, '256 slice', None, None, None, None, driver=self.driver)
        node = self.driver.create_node(name='racktest', image=image, size=size)
        self.assertEqual(node.id, '26f7fbee-8ce1-4c28-887a-bfe8e4bb10fe')
        self.assertEqual(node.name, 'racktest')
        self.assertEqual(node.extra['password'], 'racktestvJq7d3')
        self.assertEqual(node.extra['metadata']['My Server Name'], 'Apache1')

    def test_create_node_with_ex_keyname_and_ex_userdata(self):
        image = NodeImage(id=11, name='Ubuntu 8.10 (intrepid)', driver=self.driver)
        size = NodeSize(1, '256 slice', None, None, None, None, driver=self.driver)
        node = self.driver.create_node(name='racktest', image=image, size=size, ex_keyname='devstack', ex_userdata='sample data')
        self.assertEqual(node.id, '26f7fbee-8ce1-4c28-887a-bfe8e4bb10fe')
        self.assertEqual(node.name, 'racktest')
        self.assertEqual(node.extra['password'], 'racktestvJq7d3')
        self.assertEqual(node.extra['metadata']['My Server Name'], 'Apache1')
        self.assertEqual(node.extra['key_name'], 'devstack')

    def test_create_node_with_availability_zone(self):
        image = NodeImage(id=11, name='Ubuntu 8.10 (intrepid)', driver=self.driver)
        size = NodeSize(1, '256 slice', None, None, None, None, driver=self.driver)
        node = self.driver.create_node(name='racktest', image=image, size=size, ex_availability_zone='testaz')
        self.assertEqual(node.id, '26f7fbee-8ce1-4c28-887a-bfe8e4bb10fe')
        self.assertEqual(node.name, 'racktest')
        self.assertEqual(node.extra['password'], 'racktestvJq7d3')
        self.assertEqual(node.extra['metadata']['My Server Name'], 'Apache1')
        self.assertEqual(node.extra['availability_zone'], 'testaz')

    def test_create_node_with_ex_disk_config(self):
        OpenStack_1_1_MockHttp.type = 'EX_DISK_CONFIG'
        image = NodeImage(id=11, name='Ubuntu 8.10 (intrepid)', driver=self.driver)
        size = NodeSize(1, '256 slice', None, None, None, None, driver=self.driver)
        node = self.driver.create_node(name='racktest', image=image, size=size, ex_disk_config='AUTO')
        self.assertEqual(node.id, '26f7fbee-8ce1-4c28-887a-bfe8e4bb10fe')
        self.assertEqual(node.name, 'racktest')
        self.assertEqual(node.extra['disk_config'], 'AUTO')

    def test_create_node_with_ex_config_drive(self):
        OpenStack_1_1_MockHttp.type = 'EX_CONFIG_DRIVE'
        image = NodeImage(id=11, name='Ubuntu 8.10 (intrepid)', driver=self.driver)
        size = NodeSize(1, '256 slice', None, None, None, None, driver=self.driver)
        node = self.driver.create_node(name='racktest', image=image, size=size, ex_config_drive=True)
        self.assertEqual(node.id, '26f7fbee-8ce1-4c28-887a-bfe8e4bb10fe')
        self.assertEqual(node.name, 'racktest')
        self.assertTrue(node.extra['config_drive'])

    def test_create_node_from_bootable_volume(self):
        size = NodeSize(1, '256 slice', None, None, None, None, driver=self.driver)
        node = self.driver.create_node(name='racktest', size=size, ex_blockdevicemappings=[{'boot_index': 0, 'uuid': 'ee7ee330-b454-4414-8e9f-c70c558dd3af', 'source_type': 'volume', 'destination_type': 'volume', 'delete_on_termination': False}])
        self.assertEqual(node.id, '26f7fbee-8ce1-4c28-887a-bfe8e4bb10fe')
        self.assertEqual(node.name, 'racktest')
        self.assertEqual(node.extra['password'], 'racktestvJq7d3')
        self.assertEqual(node.extra['metadata']['My Server Name'], 'Apache1')

    def test_create_node_with_ex_files(self):
        OpenStack_2_0_MockHttp.type = 'EX_FILES'
        image = NodeImage(id=11, name='Ubuntu 8.10 (intrepid)', driver=self.driver)
        size = NodeSize(1, '256 slice', None, None, None, None, driver=self.driver)
        files = {'/file1': 'content1', '/file2': 'content2'}
        node = self.driver.create_node(name='racktest', image=image, size=size, ex_files=files)
        self.assertEqual(node.id, '26f7fbee-8ce1-4c28-887a-bfe8e4bb10fe')
        self.assertEqual(node.name, 'racktest')
        OpenStack_2_0_MockHttp.type = 'EX_FILES_NONE'
        node = self.driver.create_node(name='racktest', image=image, size=size)
        self.assertEqual(node.id, '26f7fbee-8ce1-4c28-887a-bfe8e4bb10fe')
        self.assertEqual(node.name, 'racktest')

    def test_destroy_node(self):
        self.assertTrue(self.node.destroy())

    def test_reboot_node(self):
        self.assertTrue(self.node.reboot())

    def test_create_volume(self):
        volume = self.driver.create_volume(1, 'test')
        self.assertEqual(volume.name, 'test')
        self.assertEqual(volume.size, 1)

    def test_create_volume_passes_location_to_request_only_if_not_none(self):
        with patch.object(self.driver.connection, 'request') as mock_request:
            self.driver.create_volume(1, 'test', location='mylocation')
            name, args, kwargs = mock_request.mock_calls[0]
            self.assertEqual(kwargs['data']['volume']['availability_zone'], 'mylocation')

    def test_create_volume_does_not_pass_location_to_request_if_none(self):
        with patch.object(self.driver.connection, 'request') as mock_request:
            self.driver.create_volume(1, 'test')
            name, args, kwargs = mock_request.mock_calls[0]
            self.assertFalse('availability_zone' in kwargs['data']['volume'])

    def test_create_volume_passes_volume_type_to_request_only_if_not_none(self):
        with patch.object(self.driver.connection, 'request') as mock_request:
            self.driver.create_volume(1, 'test', ex_volume_type='myvolumetype')
            name, args, kwargs = mock_request.mock_calls[0]
            self.assertEqual(kwargs['data']['volume']['volume_type'], 'myvolumetype')

    def test_create_volume_does_not_pass_volume_type_to_request_if_none(self):
        with patch.object(self.driver.connection, 'request') as mock_request:
            self.driver.create_volume(1, 'test')
            name, args, kwargs = mock_request.mock_calls[0]
            self.assertFalse('volume_type' in kwargs['data']['volume'])

    def test_destroy_volume(self):
        volume = self.driver.ex_get_volume('cd76a3a1-c4ce-40f6-9b9f-07a61508938d')
        self.assertEqual(self.driver.destroy_volume(volume), True)

    def test_attach_volume(self):
        node = self.driver.list_nodes()[0]
        volume = self.driver.ex_get_volume('cd76a3a1-c4ce-40f6-9b9f-07a61508938d')
        self.assertEqual(self.driver.attach_volume(node, volume, '/dev/sdb'), True)

    def test_attach_volume_device_auto(self):
        node = self.driver.list_nodes()[0]
        volume = self.driver.ex_get_volume('cd76a3a1-c4ce-40f6-9b9f-07a61508938d')
        OpenStack_2_0_MockHttp.type = 'DEVICE_AUTO'
        self.assertEqual(self.driver.attach_volume(node, volume, 'auto'), True)

    def test_detach_volume(self):
        node = self.driver.list_nodes()[0]
        volume = self.driver.ex_get_volume('cd76a3a1-c4ce-40f6-9b9f-07a61508938d')
        self.assertEqual(self.driver.attach_volume(node, volume, '/dev/sdb'), True)
        self.assertEqual(self.driver.detach_volume(volume), True)

    def test_ex_set_password(self):
        self.assertTrue(self.driver.ex_set_password(self.node, 'New1&53jPass'))

    def test_ex_rebuild(self):
        image = NodeImage(id=11, name='Ubuntu 8.10 (intrepid)', driver=self.driver)
        success = self.driver.ex_rebuild(self.node, image=image)
        self.assertTrue(success)

    def test_ex_rebuild_with_ex_disk_config(self):
        image = NodeImage(id=58, name='Ubuntu 10.10 (intrepid)', driver=self.driver)
        node = Node(id=12066, name=None, state=None, public_ips=None, private_ips=None, driver=self.driver)
        success = self.driver.ex_rebuild(node, image=image, ex_disk_config='MANUAL')
        self.assertTrue(success)

    def test_ex_rebuild_with_ex_config_drive(self):
        image = NodeImage(id=58, name='Ubuntu 10.10 (intrepid)', driver=self.driver)
        node = Node(id=12066, name=None, state=None, public_ips=None, private_ips=None, driver=self.driver)
        success = self.driver.ex_rebuild(node, image=image, ex_disk_config='MANUAL', ex_config_drive=True)
        self.assertTrue(success)

    def test_ex_resize(self):
        size = NodeSize(1, '256 slice', None, None, None, None, driver=self.driver)
        try:
            self.driver.ex_resize(self.node, size)
        except Exception as e:
            self.fail('An error was raised: ' + repr(e))

    def test_ex_confirm_resize(self):
        try:
            self.driver.ex_confirm_resize(self.node)
        except Exception as e:
            self.fail('An error was raised: ' + repr(e))

    def test_ex_revert_resize(self):
        try:
            self.driver.ex_revert_resize(self.node)
        except Exception as e:
            self.fail('An error was raised: ' + repr(e))

    def test_create_image(self):
        image = self.driver.create_image(self.node, 'new_image')
        self.assertEqual(image.name, 'new_image')
        self.assertEqual(image.id, '4949f9ee-2421-4c81-8b49-13119446008b')

    def test_ex_set_server_name(self):
        old_node = Node(id='12064', name=None, state=None, public_ips=None, private_ips=None, driver=self.driver)
        new_node = self.driver.ex_set_server_name(old_node, 'Bob')
        self.assertEqual('Bob', new_node.name)

    def test_ex_set_metadata(self):
        old_node = Node(id='12063', name=None, state=None, public_ips=None, private_ips=None, driver=self.driver)
        metadata = {'Image Version': '2.1', 'Server Label': 'Web Head 1'}
        returned_metadata = self.driver.ex_set_metadata(old_node, metadata)
        self.assertEqual(metadata, returned_metadata)

    def test_ex_get_metadata(self):
        node = Node(id='12063', name=None, state=None, public_ips=None, private_ips=None, driver=self.driver)
        metadata = {'Image Version': '2.1', 'Server Label': 'Web Head 1'}
        returned_metadata = self.driver.ex_get_metadata(node)
        self.assertEqual(metadata, returned_metadata)

    def test_ex_update_node(self):
        old_node = Node(id='12064', name=None, state=None, public_ips=None, private_ips=None, driver=self.driver)
        new_node = self.driver.ex_update_node(old_node, name='Bob')
        self.assertTrue(new_node)
        self.assertEqual('Bob', new_node.name)
        self.assertEqual('50.57.94.30', new_node.public_ips[0])

    def test_ex_get_node_details(self):
        node_id = '12064'
        node = self.driver.ex_get_node_details(node_id)
        self.assertEqual(node.id, '12064')
        self.assertEqual(node.name, 'lc-test')

    def test_ex_get_node_details_returns_none_if_node_does_not_exist(self):
        node = self.driver.ex_get_node_details('does-not-exist')
        self.assertTrue(node is None)

    def test_ex_get_node_details_microversion_2_47(self):
        node_id = '12064247'
        node = self.driver.ex_get_node_details(node_id)
        self.assertEqual(node.id, '12064247')
        self.assertEqual(node.name, 'lc-test')
        self.assertEqual(node.extra['flavor_details']['vcpus'], 2)

    def test_ex_get_size(self):
        size_id = '7'
        size = self.driver.ex_get_size(size_id)
        self.assertEqual(size.id, size_id)
        self.assertEqual(size.name, '15.5GB slice')

    def test_ex_get_size_extra_specs(self):
        size_id = '7'
        extra_specs = self.driver.ex_get_size_extra_specs(size_id)
        self.assertEqual(extra_specs, {'hw:cpu_policy': 'shared', 'hw:numa_nodes': '1'})

    def test_get_image(self):
        image_id = '13'
        image = self.driver.get_image(image_id)
        self.assertEqual(image.id, image_id)
        self.assertEqual(image.name, 'Windows 2008 SP2 x86 (B24)')
        self.assertIsNone(image.extra['serverId'])
        self.assertEqual(image.extra['minDisk'], '5')
        self.assertEqual(image.extra['minRam'], '256')
        self.assertIsNone(image.extra['visibility'])

    def test_delete_image(self):
        image = NodeImage(id='26365521-8c62-11f9-2c33-283d153ecc3a', name='My Backup', driver=self.driver)
        result = self.driver.delete_image(image)
        self.assertTrue(result)

    def test_extract_image_id_from_url(self):
        url = 'http://127.0.0.1/v1.1/68/images/1d4a8ea9-aae7-4242-a42d-5ff4702f2f14'
        url_two = 'http://127.0.0.1/v1.1/68/images/13'
        image_id = self.driver._extract_image_id_from_url(url)
        image_id_two = self.driver._extract_image_id_from_url(url_two)
        self.assertEqual(image_id, '1d4a8ea9-aae7-4242-a42d-5ff4702f2f14')
        self.assertEqual(image_id_two, '13')

    def test_ex_rescue_with_password(self):
        node = Node(id=12064, name=None, state=None, public_ips=None, private_ips=None, driver=self.driver)
        n = self.driver.ex_rescue(node, 'foo')
        self.assertEqual(n.extra['password'], 'foo')

    def test_ex_rescue_no_password(self):
        node = Node(id=12064, name=None, state=None, public_ips=None, private_ips=None, driver=self.driver)
        n = self.driver.ex_rescue(node)
        self.assertEqual(n.extra['password'], 'foo')

    def test_ex_unrescue(self):
        node = Node(id=12064, name=None, state=None, public_ips=None, private_ips=None, driver=self.driver)
        result = self.driver.ex_unrescue(node)
        self.assertTrue(result)

    def test_ex_get_node_security_groups(self):
        node = Node(id='1c01300f-ef97-4937-8f03-ac676d6234be', name=None, state=None, public_ips=None, private_ips=None, driver=self.driver)
        security_groups = self.driver.ex_get_node_security_groups(node)
        self.assertEqual(len(security_groups), 2, 'Wrong security groups count')
        security_group = security_groups[1]
        self.assertEqual(security_group.id, 4)
        self.assertEqual(security_group.tenant_id, '68')
        self.assertEqual(security_group.name, 'ftp')
        self.assertEqual(security_group.description, 'FTP Client-Server - Open 20-21 ports')
        self.assertEqual(security_group.rules[0].id, 1)
        self.assertEqual(security_group.rules[0].parent_group_id, 4)
        self.assertEqual(security_group.rules[0].ip_protocol, 'tcp')
        self.assertEqual(security_group.rules[0].from_port, 20)
        self.assertEqual(security_group.rules[0].to_port, 21)
        self.assertEqual(security_group.rules[0].ip_range, '0.0.0.0/0')

    def test_ex_list_security_groups(self):
        security_groups = self.driver.ex_list_security_groups()
        self.assertEqual(len(security_groups), 2, 'Wrong security groups count')
        security_group = security_groups[1]
        self.assertEqual(security_group.id, 4)
        self.assertEqual(security_group.tenant_id, '68')
        self.assertEqual(security_group.name, 'ftp')
        self.assertEqual(security_group.description, 'FTP Client-Server - Open 20-21 ports')
        self.assertEqual(security_group.rules[0].id, 1)
        self.assertEqual(security_group.rules[0].parent_group_id, 4)
        self.assertEqual(security_group.rules[0].ip_protocol, 'tcp')
        self.assertEqual(security_group.rules[0].from_port, 20)
        self.assertEqual(security_group.rules[0].to_port, 21)
        self.assertEqual(security_group.rules[0].ip_range, '0.0.0.0/0')

    def test_ex_create_security_group(self):
        name = 'test'
        description = 'Test Security Group'
        security_group = self.driver.ex_create_security_group(name, description)
        self.assertEqual(security_group.id, 6)
        self.assertEqual(security_group.tenant_id, '68')
        self.assertEqual(security_group.name, name)
        self.assertEqual(security_group.description, description)
        self.assertEqual(len(security_group.rules), 0)

    def test_ex_delete_security_group(self):
        security_group = OpenStackSecurityGroup(id=6, tenant_id=None, name=None, description=None, driver=self.driver)
        result = self.driver.ex_delete_security_group(security_group)
        self.assertTrue(result)

    def test_ex_create_security_group_rule(self):
        security_group = OpenStackSecurityGroup(id=6, tenant_id=None, name=None, description=None, driver=self.driver)
        security_group_rule = self.driver.ex_create_security_group_rule(security_group, 'tcp', 14, 16, '0.0.0.0/0')
        self.assertEqual(security_group_rule.id, 2)
        self.assertEqual(security_group_rule.parent_group_id, 6)
        self.assertEqual(security_group_rule.ip_protocol, 'tcp')
        self.assertEqual(security_group_rule.from_port, 14)
        self.assertEqual(security_group_rule.to_port, 16)
        self.assertEqual(security_group_rule.ip_range, '0.0.0.0/0')
        self.assertIsNone(security_group_rule.tenant_id)

    def test_ex_delete_security_group_rule(self):
        security_group_rule = OpenStackSecurityGroupRule(id=2, parent_group_id=None, ip_protocol=None, from_port=None, to_port=None, driver=self.driver)
        result = self.driver.ex_delete_security_group_rule(security_group_rule)
        self.assertTrue(result)

    def test_list_key_pairs(self):
        keypairs = self.driver.list_key_pairs()
        self.assertEqual(len(keypairs), 2, 'Wrong keypairs count')
        keypair = keypairs[1]
        self.assertEqual(keypair.name, 'key2')
        self.assertEqual(keypair.fingerprint, '5d:66:33:ae:99:0f:fb:cb:86:f2:bc:ae:53:99:b6:ed')
        self.assertTrue(len(keypair.public_key) > 10)
        self.assertIsNone(keypair.private_key)

    def test_get_key_pair(self):
        key_pair = self.driver.get_key_pair(name='test-key-pair')
        self.assertEqual(key_pair.name, 'test-key-pair')

    def test_get_key_pair_doesnt_exist(self):
        self.assertRaises(KeyPairDoesNotExistError, self.driver.get_key_pair, name='doesnt-exist')

    def test_create_key_pair(self):
        name = 'key0'
        keypair = self.driver.create_key_pair(name=name)
        self.assertEqual(keypair.name, name)
        self.assertEqual(keypair.fingerprint, '80:f8:03:a7:8e:c1:c3:b1:7e:c5:8c:50:04:5e:1c:5b')
        self.assertTrue(len(keypair.public_key) > 10)
        self.assertTrue(len(keypair.private_key) > 10)

    def test_import_key_pair_from_file(self):
        name = 'key3'
        path = os.path.join(os.path.dirname(__file__), 'fixtures', 'misc', 'test_rsa.pub')
        with open(path) as fp:
            pub_key = fp.read()
        keypair = self.driver.import_key_pair_from_file(name=name, key_file_path=path)
        self.assertEqual(keypair.name, name)
        self.assertEqual(keypair.fingerprint, '97:10:a6:e7:92:65:7e:69:fe:e6:81:8f:39:3c:8f:5a')
        self.assertEqual(keypair.public_key, pub_key)
        self.assertIsNone(keypair.private_key)

    def test_import_key_pair_from_string(self):
        name = 'key3'
        path = os.path.join(os.path.dirname(__file__), 'fixtures', 'misc', 'test_rsa.pub')
        with open(path) as fp:
            pub_key = fp.read()
        keypair = self.driver.import_key_pair_from_string(name=name, key_material=pub_key)
        self.assertEqual(keypair.name, name)
        self.assertEqual(keypair.fingerprint, '97:10:a6:e7:92:65:7e:69:fe:e6:81:8f:39:3c:8f:5a')
        self.assertEqual(keypair.public_key, pub_key)
        self.assertIsNone(keypair.private_key)

    def test_delete_key_pair(self):
        keypair = OpenStackKeyPair(name='key1', fingerprint=None, public_key=None, driver=self.driver)
        result = self.driver.delete_key_pair(key_pair=keypair)
        self.assertTrue(result)

    def test_ex_list_floating_ip_pools(self):
        ret = self.driver.ex_list_floating_ip_pools()
        self.assertEqual(ret[0].name, 'public')
        self.assertEqual(ret[1].name, 'foobar')

    def test_ex_attach_floating_ip_to_node(self):
        image = NodeImage(id=11, name='Ubuntu 8.10 (intrepid)', driver=self.driver)
        size = NodeSize(1, '256 slice', None, None, None, None, driver=self.driver)
        node = self.driver.create_node(name='racktest', image=image, size=size)
        node.id = 4242
        ip = '42.42.42.42'
        self.assertTrue(self.driver.ex_attach_floating_ip_to_node(node, ip))

    def test_detach_floating_ip_from_node(self):
        image = NodeImage(id=11, name='Ubuntu 8.10 (intrepid)', driver=self.driver)
        size = NodeSize(1, '256 slice', None, None, None, None, driver=self.driver)
        node = self.driver.create_node(name='racktest', image=image, size=size)
        node.id = 4242
        ip = '42.42.42.42'
        self.assertTrue(self.driver.ex_detach_floating_ip_from_node(node, ip))

    def test_OpenStack_1_1_FloatingIpPool_list_floating_ips(self):
        pool = OpenStack_1_1_FloatingIpPool('foo', self.driver.connection)
        ret = pool.list_floating_ips()
        self.assertEqual(ret[0].id, '09ea1784-2f81-46dc-8c91-244b4df75bde')
        self.assertEqual(ret[0].pool, pool)
        self.assertEqual(ret[0].ip_address, '10.3.1.42')
        self.assertIsNone(ret[0].node_id)
        self.assertEqual(ret[1].id, '04c5336a-0629-4694-ba30-04b0bdfa88a4')
        self.assertEqual(ret[1].pool, pool)
        self.assertEqual(ret[1].ip_address, '10.3.1.1')
        self.assertEqual(ret[1].node_id, 'fcfc96da-19e2-40fd-8497-f29da1b21143')

    def test_OpenStack_1_1_FloatingIpPool_get_floating_ip(self):
        pool = OpenStack_1_1_FloatingIpPool('foo', self.driver.connection)
        ret = pool.get_floating_ip('10.3.1.42')
        self.assertEqual(ret.id, '09ea1784-2f81-46dc-8c91-244b4df75bde')
        self.assertEqual(ret.pool, pool)
        self.assertEqual(ret.ip_address, '10.3.1.42')
        self.assertIsNone(ret.node_id)
        ret = pool.get_floating_ip('1.2.3.4')
        self.assertIsNone(ret)

    def test_OpenStack_1_1_FloatingIpPool_create_floating_ip(self):
        pool = OpenStack_1_1_FloatingIpPool('foo', self.driver.connection)
        ret = pool.create_floating_ip()
        self.assertEqual(ret.id, '09ea1784-2f81-46dc-8c91-244b4df75bde')
        self.assertEqual(ret.pool, pool)
        self.assertEqual(ret.ip_address, '10.3.1.42')
        self.assertIsNone(ret.node_id)

    def test_OpenStack_1_1_FloatingIpPool_delete_floating_ip(self):
        pool = OpenStack_1_1_FloatingIpPool('foo', self.driver.connection)
        ip = OpenStack_1_1_FloatingIpAddress('foo-bar-id', '42.42.42.42', pool)
        self.assertTrue(pool.delete_floating_ip(ip))

    def test_OpenStack_1_1_FloatingIpAddress_delete(self):
        pool = OpenStack_1_1_FloatingIpPool('foo', self.driver.connection)
        pool.delete_floating_ip = Mock()
        ip = OpenStack_1_1_FloatingIpAddress('foo-bar-id', '42.42.42.42', pool)
        ip.pool.delete_floating_ip()
        self.assertEqual(pool.delete_floating_ip.call_count, 1)

    def test_OpenStack_2_FloatingIpPool_list_floating_ips(self):
        pool = OpenStack_2_FloatingIpPool(1, 'foo', self.driver.connection)
        ret = pool.list_floating_ips()
        self.assertEqual(ret[0].id, '09ea1784-2f81-46dc-8c91-244b4df75bde')
        self.assertEqual(ret[0].get_pool(), pool)
        self.assertEqual(ret[0].ip_address, '10.3.1.42')
        self.assertEqual(ret[0].get_node_id(), None)
        self.assertEqual(ret[1].id, '04c5336a-0629-4694-ba30-04b0bdfa88a4')
        self.assertEqual(ret[1].get_pool(), pool)
        self.assertEqual(ret[1].ip_address, '10.3.1.1')
        self.assertEqual(ret[1].get_node_id(), 'fcfc96da-19e2-40fd-8497-f29da1b21143')
        self.assertEqual(ret[2].id, '123c5336a-0629-4694-ba30-04b0bdfa88a4')
        self.assertEqual(ret[2].get_pool(), pool)
        self.assertEqual(ret[2].ip_address, '10.3.1.2')
        self.assertEqual(ret[2].get_node_id(), 'cb4fba64-19e2-40fd-8497-f29da1b21143')
        self.assertEqual(ret[3].id, '123c5336a-0629-4694-ba30-04b0bdfa88a4')
        self.assertEqual(ret[3].get_pool(), pool)
        self.assertEqual(ret[3].ip_address, '10.3.1.3')
        self.assertEqual(ret[3].get_node_id(), 'cb4fba64-19e2-40fd-8497-f29da1b21143')

    def test_OpenStack_2_FloatingIpPool_get_floating_ip(self):
        pool = OpenStack_2_FloatingIpPool(1, 'foo', self.driver.connection)
        ret = pool.get_floating_ip('10.3.1.42')
        self.assertEqual(ret.id, '09ea1784-2f81-46dc-8c91-244b4df75bde')
        self.assertEqual(ret.pool, pool)
        self.assertEqual(ret.ip_address, '10.3.1.42')
        self.assertEqual(ret.node_id, None)

    def test_OpenStack_2_FloatingIpPool_create_floating_ip(self):
        pool = OpenStack_2_FloatingIpPool(1, 'foo', self.driver.connection)
        ret = pool.create_floating_ip()
        self.assertEqual(ret.id, '09ea1784-2f81-46dc-8c91-244b4df75bde')
        self.assertEqual(ret.pool, pool)
        self.assertEqual(ret.ip_address, '10.3.1.42')
        self.assertEqual(ret.node_id, None)

    def test_OpenStack_2_FloatingIpPool_delete_floating_ip(self):
        pool = OpenStack_2_FloatingIpPool(1, 'foo', self.driver.connection)
        ip = OpenStack_1_1_FloatingIpAddress('foo-bar-id', '42.42.42.42', pool)
        self.assertTrue(pool.delete_floating_ip(ip))

    def test_OpenStack_2_FloatingIpAddress_delete(self):
        pool = OpenStack_2_FloatingIpPool(1, 'foo', self.driver.connection)
        pool.delete_floating_ip = Mock()
        ip = OpenStack_1_1_FloatingIpAddress('foo-bar-id', '42.42.42.42', pool)
        ip.pool.delete_floating_ip()
        self.assertEqual(pool.delete_floating_ip.call_count, 1)

    def test_ex_get_metadata_for_node(self):
        image = NodeImage(id=11, name='Ubuntu 8.10 (intrepid)', driver=self.driver)
        size = NodeSize(1, '256 slice', None, None, None, None, driver=self.driver)
        node = self.driver.create_node(name='foo', image=image, size=size)
        metadata = self.driver.ex_get_metadata_for_node(node)
        self.assertEqual(metadata['My Server Name'], 'Apache1')
        self.assertEqual(len(metadata), 1)

    def test_ex_pause_node(self):
        node = Node(id='12063', name=None, state=None, public_ips=None, private_ips=None, driver=self.driver)
        ret = self.driver.ex_pause_node(node)
        self.assertTrue(ret is True)

    def test_ex_unpause_node(self):
        node = Node(id='12063', name=None, state=None, public_ips=None, private_ips=None, driver=self.driver)
        ret = self.driver.ex_unpause_node(node)
        self.assertTrue(ret is True)

    def test_ex_stop_node(self):
        node = Node(id='12063', name=None, state=None, public_ips=None, private_ips=None, driver=self.driver)
        ret = self.driver.ex_stop_node(node)
        self.assertTrue(ret is True)

    def test_ex_start_node(self):
        node = Node(id='12063', name=None, state=None, public_ips=None, private_ips=None, driver=self.driver)
        ret = self.driver.ex_start_node(node)
        self.assertTrue(ret is True)

    def test_ex_suspend_node(self):
        node = Node(id='12063', name=None, state=None, public_ips=None, private_ips=None, driver=self.driver)
        ret = self.driver.ex_suspend_node(node)
        self.assertTrue(ret is True)

    def test_ex_resume_node(self):
        node = Node(id='12063', name=None, state=None, public_ips=None, private_ips=None, driver=self.driver)
        ret = self.driver.ex_resume_node(node)
        self.assertTrue(ret is True)

    def test_ex_get_console_output(self):
        node = Node(id='12086', name=None, state=None, public_ips=None, private_ips=None, driver=self.driver)
        resp = self.driver.ex_get_console_output(node)
        expected_output = 'FAKE CONSOLE OUTPUT\nANOTHER\nLAST LINE'
        self.assertEqual(resp['output'], expected_output)

    def test_ex_list_snapshots(self):
        if self.driver_type.type == 'rackspace':
            self.conn_class.type = 'RACKSPACE'
        snapshots = self.driver.ex_list_snapshots()
        self.assertEqual(len(snapshots), 3)
        self.assertEqual(snapshots[0].created, datetime.datetime(2012, 2, 29, 3, 50, 7, tzinfo=UTC))
        self.assertEqual(snapshots[0].extra['created'], '2012-02-29T03:50:07Z')
        self.assertEqual(snapshots[0].extra['name'], 'snap-001')
        self.assertEqual(snapshots[0].name, 'snap-001')
        self.assertEqual(snapshots[0].state, VolumeSnapshotState.AVAILABLE)
        assert snapshots[2].created is None

    def test_ex_get_snapshot(self):
        if self.driver_type.type == 'rackspace':
            self.conn_class.type = 'RACKSPACE'
        snapshot = self.driver.ex_get_snapshot('3fbbcccf-d058-4502-8844-6feeffdf4cb5')
        self.assertEqual(snapshot.created, datetime.datetime(2012, 2, 29, 3, 50, 7, tzinfo=UTC))
        self.assertEqual(snapshot.extra['created'], '2012-02-29T03:50:07Z')
        self.assertEqual(snapshot.extra['name'], 'snap-001')
        self.assertEqual(snapshot.name, 'snap-001')
        self.assertEqual(snapshot.state, VolumeSnapshotState.AVAILABLE)

    def test_list_volume_snapshots(self):
        volume = self.driver.list_volumes()[0]
        if self.driver_type.type == 'rackspace':
            self.conn_class.type = 'RACKSPACE'
        snapshots = self.driver.list_volume_snapshots(volume)
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0].id, '4fbbdccf-e058-6502-8844-6feeffdf4cb5')

    def test_create_volume_snapshot(self):
        volume = self.driver.list_volumes()[0]
        if self.driver_type.type == 'rackspace':
            self.conn_class.type = 'RACKSPACE'
        ret = self.driver.create_volume_snapshot(volume, 'Test Volume', ex_description='This is a test', ex_force=True)
        self.assertEqual(ret.id, '3fbbcccf-d058-4502-8844-6feeffdf4cb5')

    def test_ex_create_snapshot(self):
        volume = self.driver.list_volumes()[0]
        if self.driver_type.type == 'rackspace':
            self.conn_class.type = 'RACKSPACE'
        ret = self.driver.ex_create_snapshot(volume, 'Test Volume', description='This is a test', force=True)
        self.assertEqual(ret.id, '3fbbcccf-d058-4502-8844-6feeffdf4cb5')

    def test_ex_create_snapshot_does_not_post_optional_parameters_if_none(self):
        volume = self.driver.list_volumes()[0]
        with patch.object(self.driver, '_to_snapshot'):
            with patch.object(self.driver.connection, 'request') as mock_request:
                self.driver.create_volume_snapshot(volume, name=None, ex_description=None, ex_force=True)
        name, args, kwargs = mock_request.mock_calls[0]
        self.assertFalse('display_name' in kwargs['data']['snapshot'])
        self.assertFalse('display_description' in kwargs['data']['snapshot'])

    def test_destroy_volume_snapshot(self):
        if self.driver_type.type == 'rackspace':
            self.conn_class.type = 'RACKSPACE'
        snapshot = self.driver.ex_list_snapshots()[0]
        ret = self.driver.destroy_volume_snapshot(snapshot)
        self.assertTrue(ret)

    def test_ex_delete_snapshot(self):
        if self.driver_type.type == 'rackspace':
            self.conn_class.type = 'RACKSPACE'
        snapshot = self.driver.ex_list_snapshots()[0]
        ret = self.driver.ex_delete_snapshot(snapshot)
        self.assertTrue(ret)