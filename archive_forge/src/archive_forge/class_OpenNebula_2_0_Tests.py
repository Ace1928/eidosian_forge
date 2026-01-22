import sys
import unittest
import libcloud.compute.drivers.opennebula
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeState
from libcloud.test.secrets import OPENNEBULA_PARAMS
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.opennebula import (
class OpenNebula_2_0_Tests(unittest.TestCase):
    """
    OpenNebula.org test suite for OpenNebula v2.0 through v2.2.
    """

    def setUp(self):
        """
        Setup test environment.
        """
        OpenNebulaNodeDriver.connectionCls.conn_class = OpenNebula_2_0_MockHttp
        self.driver = OpenNebulaNodeDriver(*OPENNEBULA_PARAMS + ('2.0',), host='dummy')

    def test_create_node(self):
        """
        Test create_node functionality.
        """
        image = NodeImage(id=5, name='Ubuntu 9.04 LAMP', driver=self.driver)
        size = OpenNebulaNodeSize(id=1, name='small', ram=1024, cpu=1, disk=None, bandwidth=None, price=None, driver=self.driver)
        networks = list()
        networks.append(OpenNebulaNetwork(id=5, name='Network 5', address='192.168.0.0', size=256, driver=self.driver))
        networks.append(OpenNebulaNetwork(id=15, name='Network 15', address='192.168.1.0', size=256, driver=self.driver))
        context = {'hostname': 'compute-5'}
        node = self.driver.create_node(name='Compute 5', image=image, size=size, networks=networks, context=context)
        self.assertEqual(node.id, '5')
        self.assertEqual(node.name, 'Compute 5')
        self.assertEqual(node.state, OpenNebulaNodeDriver.NODE_STATE_MAP['ACTIVE'])
        self.assertEqual(node.public_ips[0].id, '5')
        self.assertEqual(node.public_ips[0].name, 'Network 5')
        self.assertEqual(node.public_ips[0].address, '192.168.0.1')
        self.assertEqual(node.public_ips[0].size, 1)
        self.assertEqual(node.public_ips[0].extra['mac'], '02:00:c0:a8:00:01')
        self.assertEqual(node.public_ips[1].id, '15')
        self.assertEqual(node.public_ips[1].name, 'Network 15')
        self.assertEqual(node.public_ips[1].address, '192.168.1.1')
        self.assertEqual(node.public_ips[1].size, 1)
        self.assertEqual(node.public_ips[1].extra['mac'], '02:00:c0:a8:01:01')
        self.assertEqual(node.private_ips, [])
        self.assertTrue(len([s for s in self.driver.list_sizes() if s.id == node.size.id]) == 1)
        self.assertEqual(node.image.id, '5')
        self.assertEqual(node.image.name, 'Ubuntu 9.04 LAMP')
        self.assertEqual(node.image.extra['type'], 'DISK')
        self.assertEqual(node.image.extra['target'], 'hda')
        context = node.extra['context']
        self.assertEqual(context['hostname'], 'compute-5')

    def test_destroy_node(self):
        """
        Test destroy_node functionality.
        """
        node = Node(5, None, None, None, None, self.driver)
        ret = self.driver.destroy_node(node)
        self.assertTrue(ret)

    def test_list_nodes(self):
        """
        Test list_nodes functionality.
        """
        nodes = self.driver.list_nodes()
        self.assertEqual(len(nodes), 3)
        node = nodes[0]
        self.assertEqual(node.id, '5')
        self.assertEqual(node.name, 'Compute 5')
        self.assertEqual(node.state, OpenNebulaNodeDriver.NODE_STATE_MAP['ACTIVE'])
        self.assertEqual(node.public_ips[0].id, '5')
        self.assertEqual(node.public_ips[0].name, 'Network 5')
        self.assertEqual(node.public_ips[0].address, '192.168.0.1')
        self.assertEqual(node.public_ips[0].size, 1)
        self.assertEqual(node.public_ips[0].extra['mac'], '02:00:c0:a8:00:01')
        self.assertEqual(node.public_ips[1].id, '15')
        self.assertEqual(node.public_ips[1].name, 'Network 15')
        self.assertEqual(node.public_ips[1].address, '192.168.1.1')
        self.assertEqual(node.public_ips[1].size, 1)
        self.assertEqual(node.public_ips[1].extra['mac'], '02:00:c0:a8:01:01')
        self.assertEqual(node.private_ips, [])
        self.assertTrue(len([size for size in self.driver.list_sizes() if size.id == node.size.id]) == 1)
        self.assertEqual(node.size.id, '1')
        self.assertEqual(node.size.name, 'small')
        self.assertEqual(node.size.ram, 1024)
        self.assertTrue(node.size.cpu is None or isinstance(node.size.cpu, int))
        self.assertTrue(node.size.vcpu is None or isinstance(node.size.vcpu, int))
        self.assertEqual(node.size.cpu, 1)
        self.assertIsNone(node.size.vcpu)
        self.assertIsNone(node.size.disk)
        self.assertIsNone(node.size.bandwidth)
        self.assertIsNone(node.size.price)
        self.assertTrue(len([image for image in self.driver.list_images() if image.id == node.image.id]) == 1)
        self.assertEqual(node.image.id, '5')
        self.assertEqual(node.image.name, 'Ubuntu 9.04 LAMP')
        self.assertEqual(node.image.extra['type'], 'DISK')
        self.assertEqual(node.image.extra['target'], 'hda')
        context = node.extra['context']
        self.assertEqual(context['hostname'], 'compute-5')
        node = nodes[1]
        self.assertEqual(node.id, '15')
        self.assertEqual(node.name, 'Compute 15')
        self.assertEqual(node.state, OpenNebulaNodeDriver.NODE_STATE_MAP['ACTIVE'])
        self.assertEqual(node.public_ips[0].id, '5')
        self.assertEqual(node.public_ips[0].name, 'Network 5')
        self.assertEqual(node.public_ips[0].address, '192.168.0.2')
        self.assertEqual(node.public_ips[0].size, 1)
        self.assertEqual(node.public_ips[0].extra['mac'], '02:00:c0:a8:00:02')
        self.assertEqual(node.public_ips[1].id, '15')
        self.assertEqual(node.public_ips[1].name, 'Network 15')
        self.assertEqual(node.public_ips[1].address, '192.168.1.2')
        self.assertEqual(node.public_ips[1].size, 1)
        self.assertEqual(node.public_ips[1].extra['mac'], '02:00:c0:a8:01:02')
        self.assertEqual(node.private_ips, [])
        self.assertTrue(len([size for size in self.driver.list_sizes() if size.id == node.size.id]) == 1)
        self.assertEqual(node.size.id, '1')
        self.assertEqual(node.size.name, 'small')
        self.assertEqual(node.size.ram, 1024)
        self.assertTrue(node.size.cpu is None or isinstance(node.size.cpu, int))
        self.assertTrue(node.size.vcpu is None or isinstance(node.size.vcpu, int))
        self.assertEqual(node.size.cpu, 1)
        self.assertIsNone(node.size.vcpu)
        self.assertIsNone(node.size.disk)
        self.assertIsNone(node.size.bandwidth)
        self.assertIsNone(node.size.price)
        self.assertTrue(len([image for image in self.driver.list_images() if image.id == node.image.id]) == 1)
        self.assertEqual(node.image.id, '15')
        self.assertEqual(node.image.name, 'Ubuntu 9.04 LAMP')
        self.assertEqual(node.image.extra['type'], 'DISK')
        self.assertEqual(node.image.extra['target'], 'hda')
        context = node.extra['context']
        self.assertEqual(context['hostname'], 'compute-15')
        node = nodes[2]
        self.assertEqual(node.id, '25')
        self.assertEqual(node.name, 'Compute 25')
        self.assertEqual(node.state, NodeState.UNKNOWN)
        self.assertEqual(node.public_ips[0].id, '5')
        self.assertEqual(node.public_ips[0].name, 'Network 5')
        self.assertEqual(node.public_ips[0].address, '192.168.0.3')
        self.assertEqual(node.public_ips[0].size, 1)
        self.assertEqual(node.public_ips[0].extra['mac'], '02:00:c0:a8:00:03')
        self.assertEqual(node.public_ips[1].id, '15')
        self.assertEqual(node.public_ips[1].name, 'Network 15')
        self.assertEqual(node.public_ips[1].address, '192.168.1.3')
        self.assertEqual(node.public_ips[1].size, 1)
        self.assertEqual(node.public_ips[1].extra['mac'], '02:00:c0:a8:01:03')
        self.assertEqual(node.private_ips, [])
        self.assertIsNone(node.size)
        self.assertIsNone(node.image)
        context = node.extra['context']
        self.assertEqual(context, {})

    def test_list_images(self):
        """
        Test list_images functionality.
        """
        images = self.driver.list_images()
        self.assertEqual(len(images), 2)
        image = images[0]
        self.assertEqual(image.id, '5')
        self.assertEqual(image.name, 'Ubuntu 9.04 LAMP')
        self.assertEqual(image.extra['description'], 'Ubuntu 9.04 LAMP Description')
        self.assertEqual(image.extra['type'], 'OS')
        self.assertEqual(image.extra['size'], '2048')
        image = images[1]
        self.assertEqual(image.id, '15')
        self.assertEqual(image.name, 'Ubuntu 9.04 LAMP')
        self.assertEqual(image.extra['description'], 'Ubuntu 9.04 LAMP Description')
        self.assertEqual(image.extra['type'], 'OS')
        self.assertEqual(image.extra['size'], '2048')

    def test_list_sizes(self):
        """
        Test list_sizes functionality.
        """
        sizes = self.driver.list_sizes()
        self.assertEqual(len(sizes), 4)
        size = sizes[0]
        self.assertEqual(size.id, '1')
        self.assertEqual(size.name, 'small')
        self.assertEqual(size.ram, 1024)
        self.assertTrue(size.cpu is None or isinstance(size.cpu, int))
        self.assertTrue(size.vcpu is None or isinstance(size.vcpu, int))
        self.assertEqual(size.cpu, 1)
        self.assertIsNone(size.vcpu)
        self.assertIsNone(size.disk)
        self.assertIsNone(size.bandwidth)
        self.assertIsNone(size.price)
        size = sizes[1]
        self.assertEqual(size.id, '2')
        self.assertEqual(size.name, 'medium')
        self.assertEqual(size.ram, 4096)
        self.assertTrue(size.cpu is None or isinstance(size.cpu, int))
        self.assertTrue(size.vcpu is None or isinstance(size.vcpu, int))
        self.assertEqual(size.cpu, 4)
        self.assertIsNone(size.vcpu)
        self.assertIsNone(size.disk)
        self.assertIsNone(size.bandwidth)
        self.assertIsNone(size.price)
        size = sizes[2]
        self.assertEqual(size.id, '3')
        self.assertEqual(size.name, 'large')
        self.assertEqual(size.ram, 8192)
        self.assertTrue(size.cpu is None or isinstance(size.cpu, int))
        self.assertTrue(size.vcpu is None or isinstance(size.vcpu, int))
        self.assertEqual(size.cpu, 8)
        self.assertIsNone(size.vcpu)
        self.assertIsNone(size.disk)
        self.assertIsNone(size.bandwidth)
        self.assertIsNone(size.price)
        size = sizes[3]
        self.assertEqual(size.id, '4')
        self.assertEqual(size.name, 'custom')
        self.assertEqual(size.ram, 0)
        self.assertEqual(size.cpu, 0)
        self.assertIsNone(size.vcpu)
        self.assertIsNone(size.disk)
        self.assertIsNone(size.bandwidth)
        self.assertIsNone(size.price)

    def test_list_locations(self):
        """
        Test list_locations functionality.
        """
        locations = self.driver.list_locations()
        self.assertEqual(len(locations), 1)
        location = locations[0]
        self.assertEqual(location.id, '0')
        self.assertEqual(location.name, '')
        self.assertEqual(location.country, '')

    def test_ex_list_networks(self):
        """
        Test ex_list_networks functionality.
        """
        networks = self.driver.ex_list_networks()
        self.assertEqual(len(networks), 2)
        network = networks[0]
        self.assertEqual(network.id, '5')
        self.assertEqual(network.name, 'Network 5')
        self.assertEqual(network.address, '192.168.0.0')
        self.assertEqual(network.size, '256')
        network = networks[1]
        self.assertEqual(network.id, '15')
        self.assertEqual(network.name, 'Network 15')
        self.assertEqual(network.address, '192.168.1.0')
        self.assertEqual(network.size, '256')