import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node
from libcloud.test.compute import TestCaseMixin
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.maxihost import MaxihostNodeDriver
class MaxihostTest(unittest.TestCase, TestCaseMixin):

    def setUp(self):
        MaxihostNodeDriver.connectionCls.conn_class = MaxihostMockHttp
        self.driver = MaxihostNodeDriver('foo')

    def test_list_sizes(self):
        sizes = self.driver.list_sizes()
        self.assertEqual(len(sizes), 1)

    def test_list_locations(self):
        locations = self.driver.list_locations()
        self.assertEqual(len(locations), 3)

    def test_list_images(self):
        images = self.driver.list_images()
        self.assertEqual(len(images), 1)
        image = images[0]
        self.assertEqual(image.id, 'ubuntu_18_04_x64_lts')

    def test_list_key_pairs(self):
        keys = self.driver.list_key_pairs()
        self.assertEqual(len(keys), 1)
        key = keys[0]
        self.assertEqual(key.name, 'test_key')
        self.assertEqual(key.fingerprint, '77:08:a7:a5:f9:8c:e1:ab:7b:c3:d8:0c:cd:ac:8b:dd')

    def test_list_nodes(self):
        nodes = self.driver.list_nodes()
        self.assertEqual(len(nodes), 1)
        node = nodes[0]
        self.assertEqual(node.name, 'tester')

    def test_create_node_response(self):
        size = self.driver.list_sizes()[0]
        image = self.driver.list_images()[0]
        location = self.driver.list_locations()[0]
        node = self.driver.create_node(name='node-name', image=image, size=size, location=location)
        self.assertTrue(isinstance(node, Node))

    def test_destroy_node_response(self):
        node = self.driver.list_nodes()[0]
        ret = self.driver.destroy_node(node)
        self.assertTrue(ret)