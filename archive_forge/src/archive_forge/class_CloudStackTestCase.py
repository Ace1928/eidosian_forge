import os
import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlparse, parse_qsl, assertRaisesRegex
from libcloud.common.types import ProviderError
from libcloud.compute.base import NodeSize, NodeImage, NodeLocation
from libcloud.test.compute import TestCaseMixin
from libcloud.compute.types import (
from libcloud.compute.providers import get_driver
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudstack import CloudStackNodeDriver, CloudStackAffinityGroupType
class CloudStackTestCase(CloudStackCommonTestCase, unittest.TestCase):

    def test_driver_instantiation(self):
        urls = ['http://api.exoscale.ch/compute1', 'https://api.exoscale.ch/compute2', 'http://api.exoscale.ch:8888/compute3', 'https://api.exoscale.ch:8787/compute4', 'https://api.test.com/compute/endpoint']
        expected_values = [{'host': 'api.exoscale.ch', 'port': 80, 'path': '/compute1'}, {'host': 'api.exoscale.ch', 'port': 443, 'path': '/compute2'}, {'host': 'api.exoscale.ch', 'port': 8888, 'path': '/compute3'}, {'host': 'api.exoscale.ch', 'port': 8787, 'path': '/compute4'}, {'host': 'api.test.com', 'port': 443, 'path': '/compute/endpoint'}]
        cls = get_driver(Provider.CLOUDSTACK)
        for url, expected in zip(urls, expected_values):
            driver = cls('key', 'secret', url=url)
            self.assertEqual(driver.host, expected['host'])
            self.assertEqual(driver.path, expected['path'])
            self.assertEqual(driver.connection.port, expected['port'])

    def test_user_must_provide_host_and_path_or_url(self):
        expected_msg = 'When instantiating CloudStack driver directly you also need to provide url or host and path argument'
        cls = get_driver(Provider.CLOUDSTACK)
        assertRaisesRegex(self, Exception, expected_msg, cls, 'key', 'secret')
        try:
            cls('key', 'secret', True, 'localhost', '/path')
        except Exception:
            self.fail('host and path provided but driver raised an exception')
        try:
            cls('key', 'secret', url='https://api.exoscale.ch/compute')
        except Exception:
            self.fail('url provided but driver raised an exception')

    def test_restore(self):
        template = NodeImage('aaa-bbb-ccc-ddd', 'fake-img', None)
        node = self.driver.list_nodes()[0]
        res = node.ex_restore(template=template)
        self.assertEqual(res, template.id)

    def test_change_offerings(self):
        offering = NodeSize('eee-fff-ggg-hhh', 'fake-size', 1, 4, 5, 0.1, None)
        node = self.driver.list_nodes()[0]
        res = node.ex_change_node_size(offering=offering)
        self.assertEqual(res, offering.id)