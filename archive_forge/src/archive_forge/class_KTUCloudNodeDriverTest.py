import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib, urlparse, parse_qsl
from libcloud.test.compute import TestCaseMixin
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.ktucloud import KTUCloudNodeDriver
class KTUCloudNodeDriverTest(unittest.TestCase, TestCaseMixin):

    def setUp(self):
        KTUCloudNodeDriver.connectionCls.conn_class = KTUCloudStackMockHttp
        self.driver = KTUCloudNodeDriver('apikey', 'secret', path='/test/path', host='api.dummy.com')
        self.driver.path = '/test/path'
        self.driver.type = -1
        KTUCloudStackMockHttp.fixture_tag = 'default'
        self.driver.connection.poll_interval = 0.0

    def test_create_node_immediate_failure(self):
        size = self.driver.list_sizes()[0]
        image = self.driver.list_images()[0]
        KTUCloudStackMockHttp.fixture_tag = 'deployfail'
        self.assertRaises(Exception, self.driver.create_node, name='node-name', image=image, size=size)

    def test_create_node_delayed_failure(self):
        size = self.driver.list_sizes()[0]
        image = self.driver.list_images()[0]
        KTUCloudStackMockHttp.fixture_tag = 'deployfail2'
        self.assertRaises(Exception, self.driver.create_node, name='node-name', image=image, size=size)

    def test_list_images_no_images_available(self):
        KTUCloudStackMockHttp.fixture_tag = 'notemplates'
        images = self.driver.list_images()
        self.assertEqual(0, len(images))

    def test_list_images_available(self):
        images = self.driver.list_images()
        self.assertEqual(112, len(images))

    def test_list_sizes_available(self):
        sizes = self.driver.list_sizes()
        self.assertEqual(112, len(sizes))

    def test_list_sizes_nodisk(self):
        KTUCloudStackMockHttp.fixture_tag = 'nodisk'
        sizes = self.driver.list_sizes()
        self.assertEqual(2, len(sizes))
        check = False
        size = sizes[1]
        if size.id == KTUCloudNodeDriver.EMPTY_DISKOFFERINGID:
            check = True
        self.assertTrue(check)