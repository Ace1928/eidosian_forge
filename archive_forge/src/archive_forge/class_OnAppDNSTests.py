import sys
import unittest
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.dns.types import RecordType
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_ONAPP
from libcloud.common.exceptions import BaseHTTPError
from libcloud.dns.drivers.onapp import OnAppDNSDriver
from libcloud.test.file_fixtures import DNSFileFixtures
class OnAppDNSTests(LibcloudTestCase):

    def setUp(self):
        OnAppDNSDriver.connectionCls.conn_class = OnAppDNSMockHttp
        OnAppDNSMockHttp.type = None
        self.driver = OnAppDNSDriver(*DNS_PARAMS_ONAPP)

    def assertHasKeys(self, dictionary, keys):
        for key in keys:
            self.assertTrue(key in dictionary, 'key "%s" not in dictionary' % key)

    def test_list_record_types(self):
        record_types = self.driver.list_record_types()
        self.assertEqual(len(record_types), 8)
        self.assertTrue(RecordType.A in record_types)
        self.assertTrue(RecordType.AAAA in record_types)
        self.assertTrue(RecordType.CNAME in record_types)
        self.assertTrue(RecordType.MX in record_types)
        self.assertTrue(RecordType.NS in record_types)
        self.assertTrue(RecordType.SOA in record_types)
        self.assertTrue(RecordType.SRV in record_types)
        self.assertTrue(RecordType.TXT in record_types)

    def test_list_zones_success(self):
        zones = self.driver.list_zones()
        self.assertEqual(len(zones), 2)
        zone1 = zones[0]
        self.assertEqual(zone1.id, '1')
        self.assertEqual(zone1.type, 'master')
        self.assertEqual(zone1.domain, 'example.com')
        self.assertEqual(zone1.ttl, 1200)
        self.assertHasKeys(zone1.extra, ['user_id', 'cdn_reference', 'created_at', 'updated_at'])
        zone2 = zones[1]
        self.assertEqual(zone2.id, '2')
        self.assertEqual(zone2.type, 'master')
        self.assertEqual(zone2.domain, 'example.net')
        self.assertEqual(zone2.ttl, 1200)
        self.assertHasKeys(zone2.extra, ['user_id', 'cdn_reference', 'created_at', 'updated_at'])

    def test_get_zone_success(self):
        zone1 = self.driver.get_zone(zone_id='1')
        self.assertEqual(zone1.id, '1')
        self.assertEqual(zone1.type, 'master')
        self.assertEqual(zone1.domain, 'example.com')
        self.assertEqual(zone1.ttl, 1200)
        self.assertHasKeys(zone1.extra, ['user_id', 'cdn_reference', 'created_at', 'updated_at'])

    def test_get_zone_not_found(self):
        OnAppDNSMockHttp.type = 'NOT_FOUND'
        try:
            self.driver.get_zone(zone_id='3')
        except BaseHTTPError:
            self.assertRaises(Exception)

    def test_create_zone_success(self):
        OnAppDNSMockHttp.type = 'CREATE'
        zone = self.driver.create_zone(domain='example.com')
        self.assertEqual(zone.id, '1')
        self.assertEqual(zone.domain, 'example.com')
        self.assertEqual(zone.ttl, 1200)
        self.assertEqual(zone.type, 'master')
        self.assertHasKeys(zone.extra, ['user_id', 'cdn_reference', 'created_at', 'updated_at'])

    def test_delete_zone(self):
        zone = self.driver.get_zone(zone_id='1')
        OnAppDNSMockHttp.type = 'DELETE'
        self.assertTrue(self.driver.delete_zone(zone))

    def test_list_records_success(self):
        zone = self.driver.get_zone(zone_id='1')
        records = self.driver.list_records(zone=zone)
        self.assertEqual(len(records), 5)
        record1 = records[0]
        self.assertEqual(record1.id, '111222')
        self.assertEqual(record1.name, '@')
        self.assertEqual(record1.type, RecordType.A)
        self.assertEqual(record1.ttl, 3600)
        self.assertEqual(record1.data['ip'], '123.156.189.1')
        record2 = records[2]
        self.assertEqual(record2.id, '111224')
        self.assertEqual(record2.name, 'mail')
        self.assertEqual(record1.ttl, 3600)
        self.assertEqual(record2.type, RecordType.CNAME)
        self.assertEqual(record2.data['hostname'], 'examplemail.com')
        record3 = records[4]
        self.assertEqual(record3.id, '111226')
        self.assertEqual(record3.name, '@')
        self.assertEqual(record3.type, RecordType.MX)
        self.assertEqual(record3.data['hostname'], 'mx2.examplemail.com')

    def test_get_record_success(self):
        record = self.driver.get_record(zone_id='1', record_id='123')
        self.assertEqual(record.id, '123')
        self.assertEqual(record.name, '@')
        self.assertEqual(record.type, RecordType.A)
        self.assertEqual(record.data['ip'], '123.156.189.1')

    def test_create_record_success(self):
        zone = self.driver.get_zone(zone_id='1')
        OnAppDNSMockHttp.type = 'CREATE'
        record = self.driver.create_record(name='blog', zone=zone, type=RecordType.A, data='123.156.189.2')
        self.assertEqual(record.id, '111227')
        self.assertEqual(record.name, 'blog')
        self.assertEqual(record.type, RecordType.A)
        self.assertEqual(record.data['ip'], '123.156.189.2')
        self.assertEqual(record.data['ttl'], 3600)

    def test_update_record_success(self):
        record = self.driver.get_record(zone_id='1', record_id='123')
        OnAppDNSMockHttp.type = 'UPDATE'
        extra = {'ttl': 4500}
        record1 = self.driver.update_record(record=record, name='@', type=record.type, data='123.156.189.2', extra=extra)
        self.assertEqual(record.data['ip'], '123.156.189.1')
        self.assertEqual(record.ttl, 3600)
        self.assertEqual(record1.data['ip'], '123.156.189.2')
        self.assertEqual(record1.ttl, 4500)

    def test_delete_record_success(self):
        record = self.driver.get_record(zone_id='1', record_id='123')
        OnAppDNSMockHttp.type = 'DELETE'
        status = self.driver.delete_record(record=record)
        self.assertTrue(status)