import sys
import unittest
from datetime import datetime
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import NodeImage
from libcloud.test.secrets import DIGITALOCEAN_v1_PARAMS, DIGITALOCEAN_v2_PARAMS
from libcloud.utils.iso8601 import UTC
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.common.digitalocean import DigitalOcean_v1_Error
from libcloud.compute.drivers.digitalocean import DigitalOceanNodeDriver
class DigitalOcean_v2_Tests(LibcloudTestCase):

    def setUp(self):
        DigitalOceanNodeDriver.connectionCls.conn_class = DigitalOceanMockHttp
        DigitalOceanMockHttp.type = None
        self.driver = DigitalOceanNodeDriver(*DIGITALOCEAN_v2_PARAMS)

    def test_v1_Error(self):
        self.assertRaises(DigitalOcean_v1_Error, DigitalOceanNodeDriver, *DIGITALOCEAN_v1_PARAMS, api_version='v1')

    def test_v2_uses_v1_key(self):
        self.assertRaises(InvalidCredsError, DigitalOceanNodeDriver, *DIGITALOCEAN_v1_PARAMS, api_version='v2')

    def test_authentication(self):
        DigitalOceanMockHttp.type = 'UNAUTHORIZED'
        self.assertRaises(InvalidCredsError, self.driver.list_nodes)

    def test_list_images_success(self):
        images = self.driver.list_images()
        self.assertTrue(len(images) >= 1)
        image = images[0]
        self.assertTrue(image.id is not None)
        self.assertTrue(image.name is not None)

    def test_list_sizes_success(self):
        sizes = self.driver.list_sizes()
        self.assertTrue(len(sizes) >= 1)
        size = sizes[0]
        self.assertTrue(size.id is not None)
        self.assertEqual(size.name, '512mb')
        self.assertEqual(size.ram, 512)
        size = sizes[1]
        self.assertTrue(size.id is not None)
        self.assertEqual(size.name, '1gb')
        self.assertEqual(size.ram, 1024)

    def test_list_sizes_filter_by_location_success(self):
        location = self.driver.list_locations()[1]
        sizes = self.driver.list_sizes(location=location)
        self.assertTrue(len(sizes) >= 1)
        size = sizes[0]
        self.assertTrue(size.id is not None)
        self.assertEqual(size.name, '512mb')
        self.assertTrue(location.id in size.extra['regions'])
        location = self.driver.list_locations()[1]
        location.id = 'doesntexist'
        sizes = self.driver.list_sizes(location=location)
        self.assertEqual(len(sizes), 0)

    def test_list_locations_success(self):
        locations = self.driver.list_locations()
        self.assertTrue(len(locations) == 2)
        location = locations[0]
        self.assertEqual(location.id, 'nyc1')
        self.assertEqual(location.name, 'New York 1')
        locations = self.driver.list_locations(ex_available=True)
        self.assertTrue(len(locations) == 2)
        locations = self.driver.list_locations(ex_available=False)
        self.assertTrue(len(locations) == 3)

    def test_list_nodes_success(self):
        nodes = self.driver.list_nodes()
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].name, 'ubuntu-s-1vcpu-1gb-sfo3-01')
        self.assertEqual(nodes[0].public_ips, ['128.199.13.158'])
        self.assertEqual(nodes[0].extra['image']['id'], 69463186)
        self.assertEqual(nodes[0].extra['size_slug'], 's-1vcpu-1gb')
        self.assertEqual(len(nodes[0].extra['tags']), 0)

    def test_list_nodes_fills_created_datetime(self):
        nodes = self.driver.list_nodes()
        self.assertEqual(nodes[0].created_at, datetime(2020, 10, 15, 13, 58, 22, tzinfo=UTC))

    def test_create_node_invalid_size(self):
        image = NodeImage(id='invalid', name=None, driver=self.driver)
        size = self.driver.list_sizes()[0]
        location = self.driver.list_locations()[0]
        DigitalOceanMockHttp.type = 'INVALID_IMAGE'
        expected_msg = 'You specified an invalid image for Droplet creation.' + ' \\(code: (404|HTTPStatus.NOT_FOUND)\\)'
        assertRaisesRegex(self, Exception, expected_msg, self.driver.create_node, name='test', size=size, image=image, location=location)

    def test_reboot_node_success(self):
        node = self.driver.list_nodes()[0]
        DigitalOceanMockHttp.type = 'REBOOT'
        result = self.driver.reboot_node(node)
        self.assertTrue(result)

    def test_create_image_success(self):
        node = self.driver.list_nodes()[0]
        DigitalOceanMockHttp.type = 'SNAPSHOT'
        result = self.driver.create_image(node, 'My snapshot')
        self.assertTrue(result)

    def test_get_image_success(self):
        image = self.driver.get_image(12345)
        self.assertEqual(image.name, 'My snapshot')
        self.assertEqual(image.id, '12345')
        self.assertEqual(image.extra['distribution'], 'Ubuntu')

    def test_delete_image_success(self):
        image = self.driver.get_image(12345)
        DigitalOceanMockHttp.type = 'DESTROY'
        result = self.driver.delete_image(image)
        self.assertTrue(result)

    def test_ex_power_on_node_success(self):
        node = self.driver.list_nodes()[0]
        DigitalOceanMockHttp.type = 'POWERON'
        result = self.driver.ex_power_on_node(node)
        self.assertTrue(result)

    def test_ex_shutdown_node_success(self):
        node = self.driver.list_nodes()[0]
        DigitalOceanMockHttp.type = 'SHUTDOWN'
        result = self.driver.ex_shutdown_node(node)
        self.assertTrue(result)

    def test_ex_hard_reboot_success(self):
        node = self.driver.list_nodes()[0]
        DigitalOceanMockHttp.type = 'POWERCYCLE'
        result = self.driver.ex_hard_reboot(node)
        self.assertTrue(result)

    def test_ex_rebuild_node_success(self):
        node = self.driver.list_nodes()[0]
        DigitalOceanMockHttp.type = 'REBUILD'
        result = self.driver.ex_rebuild_node(node)
        self.assertTrue(result)

    def test_ex_resize_node_success(self):
        node = self.driver.list_nodes()[0]
        size = self.driver.list_sizes()[0]
        DigitalOceanMockHttp.type = 'RESIZE'
        result = self.driver.ex_resize_node(node, size)
        self.assertTrue(result)

    def test_destroy_node_success(self):
        node = self.driver.list_nodes()[0]
        DigitalOceanMockHttp.type = 'DESTROY'
        result = self.driver.destroy_node(node)
        self.assertTrue(result)

    def test_ex_change_kernel_success(self):
        node = self.driver.list_nodes()[0]
        DigitalOceanMockHttp.type = 'KERNELCHANGE'
        result = self.driver.ex_change_kernel(node, 7515)
        self.assertTrue(result)

    def test_ex_enable_ipv6_success(self):
        node = self.driver.list_nodes()[0]
        DigitalOceanMockHttp.type = 'ENABLEIPV6'
        result = self.driver.ex_enable_ipv6(node)
        self.assertTrue(result)

    def test_ex_rename_node_success(self):
        node = self.driver.list_nodes()[0]
        DigitalOceanMockHttp.type = 'RENAME'
        result = self.driver.ex_rename_node(node, 'fedora helios')
        self.assertTrue(result)

    def test_list_key_pairs(self):
        keys = self.driver.list_key_pairs()
        self.assertEqual(len(keys), 1)
        self.assertEqual(keys[0].extra['id'], 7717)
        self.assertEqual(keys[0].name, 'test1')
        self.assertEqual(keys[0].public_key, 'ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAAAQQDGk5 example')

    def test_create_key_pair(self):
        DigitalOceanMockHttp.type = 'CREATE'
        key = self.driver.create_key_pair(name='test1', public_key='ssh-rsa AAAAB3NzaC1yc2EAAAADAQsxRiUKn example')
        self.assertEqual(key.name, 'test1')
        self.assertEqual(key.fingerprint, 'f5:d1:78:ed:28:72:5f:e1:ac:94:fd:1f:e0:a3:48:6d')

    def test_delete_key_pair(self):
        key = self.driver.list_key_pairs()[0]
        result = self.driver.delete_key_pair(key)
        self.assertTrue(result)

    def test__paginated_request_single_page(self):
        nodes = self.driver._paginated_request('/v2/droplets', 'droplets')
        self.assertEqual(nodes[0]['name'], 'ubuntu-s-1vcpu-1gb-sfo3-01')
        self.assertEqual(nodes[0]['image']['id'], 69463186)
        self.assertEqual(nodes[0]['size_slug'], 's-1vcpu-1gb')

    def test__paginated_request_two_pages(self):
        DigitalOceanMockHttp.type = 'PAGE_ONE'
        nodes = self.driver._paginated_request('/v2/droplets', 'droplets')
        self.assertEqual(len(nodes), 2)

    def test_list_volumes(self):
        volumes = self.driver.list_volumes()
        self.assertEqual(len(volumes), 1)
        volume = volumes[0]
        self.assertEqual(volume.id, '62766883-2c28-11e6-b8e6-000f53306ae1')
        self.assertEqual(volume.name, 'example')
        self.assertEqual(volume.size, 4)
        self.assertEqual(volume.driver, self.driver)

    def test_list_volumes_empty(self):
        DigitalOceanMockHttp.type = 'EMPTY'
        volumes = self.driver.list_volumes()
        self.assertEqual(len(volumes), 0)

    def test_create_volume(self):
        nyc1 = [r for r in self.driver.list_locations() if r.id == 'nyc1'][0]
        DigitalOceanMockHttp.type = 'CREATE'
        volume = self.driver.create_volume(4, 'example', nyc1)
        self.assertEqual(volume.id, '62766883-2c28-11e6-b8e6-000f53306ae1')
        self.assertEqual(volume.name, 'example')
        self.assertEqual(volume.size, 4)
        self.assertEqual(volume.driver, self.driver)

    def test_attach_volume(self):
        node = self.driver.list_nodes()[0]
        volume = self.driver.list_volumes()[0]
        DigitalOceanMockHttp.type = 'ATTACH'
        resp = self.driver.attach_volume(node, volume)
        self.assertTrue(resp)

    def test_detach_volume(self):
        volume = self.driver.list_volumes()[0]
        DigitalOceanMockHttp.type = 'DETACH'
        resp = self.driver.detach_volume(volume)
        self.assertTrue(resp)

    def test_destroy_volume(self):
        volume = self.driver.list_volumes()[0]
        DigitalOceanMockHttp.type = 'DESTROY'
        resp = self.driver.destroy_volume(volume)
        self.assertTrue(resp)

    def test_list_volume_snapshots(self):
        volume = self.driver.list_volumes()[0]
        snapshots = self.driver.list_volume_snapshots(volume)
        self.assertEqual(len(snapshots), 3)
        snapshot1, snapshot2, snapshot3 = snapshots
        self.assertEqual(snapshot1.id, 'c0def940-9324-11e6-9a56-000f533176b1')
        self.assertEqual(snapshot2.id, 'c2036724-9343-11e6-aef4-000f53315a41')
        self.assertEqual(snapshot3.id, 'd347e033-9343-11e6-9a56-000f533176b1')

    def test_create_volume_snapshot(self):
        volume = self.driver.list_volumes()[0]
        DigitalOceanMockHttp.type = 'CREATE'
        snapshot = self.driver.create_volume_snapshot(volume, 'test-snapshot')
        self.assertEqual(snapshot.id, 'c0def940-9324-11e6-9a56-000f533176b1')
        self.assertEqual(snapshot.name, 'test-snapshot')
        self.assertEqual(volume.driver, self.driver)

    def test_delete_volume_snapshot(self):
        volume = self.driver.list_volumes()[0]
        snapshot = self.driver.list_volume_snapshots(volume)[0]
        DigitalOceanMockHttp.type = 'DELETE'
        result = self.driver.delete_volume_snapshot(snapshot)
        self.assertTrue(result)

    def test_ex_get_node_details(self):
        node = self.driver.ex_get_node_details('3164444')
        self.assertEqual(node.name, 'example.com')
        self.assertEqual(node.public_ips, ['36.123.0.123'])
        self.assertEqual(node.extra['image']['id'], 12089443)
        self.assertEqual(node.extra['size_slug'], '8gb')
        self.assertEqual(len(node.extra['tags']), 2)

    def test_ex_create_floating_ip(self):
        nyc1 = [r for r in self.driver.list_locations() if r.id == 'nyc1'][0]
        floating_ip = self.driver.ex_create_floating_ip(nyc1)
        self.assertEqual(floating_ip.id, '167.138.123.111')
        self.assertEqual(floating_ip.ip_address, '167.138.123.111')
        self.assertEqual(floating_ip.extra['region']['slug'], 'nyc1')
        self.assertIsNone(floating_ip.node_id)

    def test_ex_delete_floating_ip(self):
        nyc1 = [r for r in self.driver.list_locations() if r.id == 'nyc1'][0]
        floating_ip = self.driver.ex_create_floating_ip(nyc1)
        ret = self.driver.ex_delete_floating_ip(floating_ip)
        self.assertTrue(ret)

    def test_floating_ip_can_be_deleted_by_calling_delete_on_floating_ip_object(self):
        nyc1 = [r for r in self.driver.list_locations() if r.id == 'nyc1'][0]
        floating_ip = self.driver.ex_create_floating_ip(nyc1)
        ret = floating_ip.delete()
        self.assertTrue(ret)

    def test_list_floating_ips(self):
        floating_ips = self.driver.ex_list_floating_ips()
        self.assertEqual(len(floating_ips), 2, 'Wrong floating IPs count')
        floating_ip = floating_ips[0]
        self.assertEqual(floating_ip.id, '133.166.122.204')
        self.assertEqual(floating_ip.ip_address, '133.166.122.204')
        self.assertEqual(floating_ip.extra['region']['slug'], 'ams3')
        self.assertEqual(84155775, floating_ip.node_id)

    def test_get_floating_ip(self):
        floating_ip = self.driver.ex_get_floating_ip('133.166.122.204')
        self.assertEqual(floating_ip.id, '133.166.122.204')
        self.assertEqual(floating_ip.ip_address, '133.166.122.204')
        self.assertEqual(floating_ip.extra['region']['slug'], 'ams3')
        self.assertEqual(84155775, floating_ip.node_id)

    def test_ex_attach_floating_ip_to_node(self):
        node = self.driver.list_nodes()[0]
        floating_ip = self.driver.ex_get_floating_ip('133.166.122.204')
        ret = self.driver.ex_attach_floating_ip_to_node(node, floating_ip)
        self.assertTrue(ret)

    def test_ex_detach_floating_ip_from_node(self):
        node = self.driver.list_nodes()[0]
        floating_ip = self.driver.ex_get_floating_ip('154.138.103.175')
        ret = self.driver.ex_detach_floating_ip_from_node(node, floating_ip)
        self.assertTrue(ret)