import sys
import unittest
import libcloud.compute.drivers.opennebula
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeState
from libcloud.test.secrets import OPENNEBULA_PARAMS
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.opennebula import (
class OpenNebula_1_4_Tests(unittest.TestCase):
    """
    OpenNebula.org test suite for OpenNebula v1.4.
    """

    def setUp(self):
        """
        Setup test environment.
        """
        OpenNebulaNodeDriver.connectionCls.conn_class = OpenNebula_1_4_MockHttp
        self.driver = OpenNebulaNodeDriver(*OPENNEBULA_PARAMS + ('1.4',), host='dummy')

    def test_create_node(self):
        """
        Test create_node functionality.
        """
        image = NodeImage(id=5, name='Ubuntu 9.04 LAMP', driver=self.driver)
        size = NodeSize(id=1, name='small', ram=None, disk=None, bandwidth=None, price=None, driver=self.driver)
        networks = list()
        networks.append(OpenNebulaNetwork(id=5, name='Network 5', address='192.168.0.0', size=256, driver=self.driver))
        networks.append(OpenNebulaNetwork(id=15, name='Network 15', address='192.168.1.0', size=256, driver=self.driver))
        node = self.driver.create_node(name='Compute 5', image=image, size=size, networks=networks)
        self.assertEqual(node.id, '5')
        self.assertEqual(node.name, 'Compute 5')
        self.assertEqual(node.state, OpenNebulaNodeDriver.NODE_STATE_MAP['ACTIVE'])
        self.assertIsNone(node.public_ips[0].name)
        self.assertEqual(node.public_ips[0].id, '5')
        self.assertEqual(node.public_ips[0].address, '192.168.0.1')
        self.assertEqual(node.public_ips[0].size, 1)
        self.assertIsNone(node.public_ips[1].name)
        self.assertEqual(node.public_ips[1].id, '15')
        self.assertEqual(node.public_ips[1].address, '192.168.1.1')
        self.assertEqual(node.public_ips[1].size, 1)
        self.assertEqual(node.private_ips, [])
        self.assertEqual(node.image.id, '5')
        self.assertEqual(node.image.extra['dev'], 'sda1')

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
        self.assertIsNone(node.public_ips[0].name)
        self.assertEqual(node.public_ips[0].address, '192.168.0.1')
        self.assertEqual(node.public_ips[0].size, 1)
        self.assertEqual(node.public_ips[1].id, '15')
        self.assertIsNone(node.public_ips[1].name)
        self.assertEqual(node.public_ips[1].address, '192.168.1.1')
        self.assertEqual(node.public_ips[1].size, 1)
        self.assertEqual(node.private_ips, [])
        self.assertEqual(node.image.id, '5')
        self.assertEqual(node.image.extra['dev'], 'sda1')
        node = nodes[1]
        self.assertEqual(node.id, '15')
        self.assertEqual(node.name, 'Compute 15')
        self.assertEqual(node.state, OpenNebulaNodeDriver.NODE_STATE_MAP['ACTIVE'])
        self.assertEqual(node.public_ips[0].id, '5')
        self.assertIsNone(node.public_ips[0].name)
        self.assertEqual(node.public_ips[0].address, '192.168.0.2')
        self.assertEqual(node.public_ips[0].size, 1)
        self.assertEqual(node.public_ips[1].id, '15')
        self.assertIsNone(node.public_ips[1].name)
        self.assertEqual(node.public_ips[1].address, '192.168.1.2')
        self.assertEqual(node.public_ips[1].size, 1)
        self.assertEqual(node.private_ips, [])
        self.assertEqual(node.image.id, '15')
        self.assertEqual(node.image.extra['dev'], 'sda1')
        node = nodes[2]
        self.assertEqual(node.id, '25')
        self.assertEqual(node.name, 'Compute 25')
        self.assertEqual(node.state, NodeState.UNKNOWN)
        self.assertEqual(node.public_ips[0].id, '5')
        self.assertIsNone(node.public_ips[0].name)
        self.assertEqual(node.public_ips[0].address, '192.168.0.3')
        self.assertEqual(node.public_ips[0].size, 1)
        self.assertEqual(node.public_ips[1].id, '15')
        self.assertIsNone(node.public_ips[1].name)
        self.assertEqual(node.public_ips[1].address, '192.168.1.3')
        self.assertEqual(node.public_ips[1].size, 1)
        self.assertEqual(node.private_ips, [])
        self.assertIsNone(node.image)

    def test_list_images(self):
        """
        Test list_images functionality.
        """
        images = self.driver.list_images()
        self.assertEqual(len(images), 2)
        image = images[0]
        self.assertEqual(image.id, '5')
        self.assertEqual(image.name, 'Ubuntu 9.04 LAMP')
        self.assertEqual(image.extra['size'], '2048')
        self.assertEqual(image.extra['url'], 'file:///images/ubuntu/jaunty.img')
        image = images[1]
        self.assertEqual(image.id, '15')
        self.assertEqual(image.name, 'Ubuntu 9.04 LAMP')
        self.assertEqual(image.extra['size'], '2048')
        self.assertEqual(image.extra['url'], 'file:///images/ubuntu/jaunty.img')

    def test_list_sizes(self):
        """
        Test list_sizes functionality.
        """
        sizes = self.driver.list_sizes()
        self.assertEqual(len(sizes), 3)
        size = sizes[0]
        self.assertEqual(size.id, '1')
        self.assertEqual(size.name, 'small')
        self.assertIsNone(size.ram)
        self.assertIsNone(size.disk)
        self.assertIsNone(size.bandwidth)
        self.assertIsNone(size.price)
        size = sizes[1]
        self.assertEqual(size.id, '2')
        self.assertEqual(size.name, 'medium')
        self.assertIsNone(size.ram)
        self.assertIsNone(size.disk)
        self.assertIsNone(size.bandwidth)
        self.assertIsNone(size.price)
        size = sizes[2]
        self.assertEqual(size.id, '3')
        self.assertEqual(size.name, 'large')
        self.assertIsNone(size.ram)
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

    def test_ex_node_action(self):
        """
        Test ex_node_action functionality.
        """
        node = Node(5, None, None, None, None, self.driver)
        ret = self.driver.ex_node_action(node, ACTION.STOP)
        self.assertTrue(ret)