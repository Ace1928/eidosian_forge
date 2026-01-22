import sys
import unittest
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.dns.types import RecordType
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DIGITALOCEAN_v2_PARAMS
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.digitalocean import DigitalOceanDNSDriver
class DigitalOceanDNSTests(LibcloudTestCase):

    def setUp(self):
        DigitalOceanDNSDriver.connectionCls.conn_class = DigitalOceanDNSMockHttp
        DigitalOceanDNSMockHttp.type = None
        self.driver = DigitalOceanDNSDriver(*DIGITALOCEAN_v2_PARAMS)

    def test_list_zones(self):
        zones = self.driver.list_zones()
        self.assertTrue(len(zones) >= 1)

    def test_get_zone(self):
        zone = self.driver.get_zone('testdomain')
        self.assertEqual(zone.id, 'testdomain')

    def test_get_zone_not_found(self):
        DigitalOceanDNSMockHttp.type = 'NOT_FOUND'
        self.assertRaises(Exception, self.driver.get_zone, 'testdomain')

    def test_list_records(self):
        zone = self.driver.get_zone('testdomain')
        records = self.driver.list_records(zone)
        self.assertTrue(len(records) >= 1)
        self.assertEqual(records[1].ttl, 1800)
        self.assertEqual(records[4].ttl, None)

    def test_get_record(self):
        record = self.driver.get_record('testdomain', '1234564')
        self.assertEqual(record.id, '1234564')
        self.assertEqual(record.type, RecordType.A)
        self.assertEqual(record.data, '123.45.67.89')
        self.assertEqual(record.ttl, 1800)

    def test_get_record_not_found(self):
        DigitalOceanDNSMockHttp.type = 'NOT_FOUND'
        self.assertRaises(Exception, self.driver.get_zone, 'testdomain')

    def test_create_zone(self):
        DigitalOceanDNSMockHttp.type = 'CREATE'
        zone = self.driver.create_zone('testdomain')
        self.assertEqual(zone.id, 'testdomain')

    def test_create_record(self):
        zone = self.driver.get_zone('testdomain')
        DigitalOceanDNSMockHttp.type = 'CREATE'
        record = self.driver.create_record('sub', zone, RecordType.A, '234.56.78.90', extra={'ttl': 60})
        self.assertEqual(record.id, '1234565')
        self.assertEqual(record.type, RecordType.A)
        self.assertEqual(record.data, '234.56.78.90')
        self.assertEqual(record.ttl, 60)

    def test_update_record(self):
        record = self.driver.get_record('testdomain', '1234564')
        DigitalOceanDNSMockHttp.type = 'UPDATE'
        record = self.driver.update_record(record, data='234.56.78.90', extra={'ttl': 60})
        self.assertEqual(record.id, '1234564')
        self.assertEqual(record.data, '234.56.78.90')
        self.assertEqual(record.ttl, 60)

    def test_delete_zone(self):
        zone = self.driver.get_zone('testdomain')
        DigitalOceanDNSMockHttp.type = 'DELETE'
        self.assertTrue(self.driver.delete_zone(zone))

    def test_delete_record(self):
        record = self.driver.get_record('testdomain', '1234564')
        DigitalOceanDNSMockHttp.type = 'DELETE'
        self.assertTrue(self.driver.delete_record(record))