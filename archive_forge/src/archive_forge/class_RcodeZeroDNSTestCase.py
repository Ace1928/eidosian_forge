import sys
import json
import unittest
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, ZoneAlreadyExistsError
from libcloud.utils.py3 import httplib
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.rcodezero import RcodeZeroDNSDriver
class RcodeZeroDNSTestCase(LibcloudTestCase):

    def setUp(self):
        RcodeZeroDNSDriver.connectionCls.conn_class = RcodeZeroDNSMockHttp
        RcodeZeroDNSMockHttp.type = None
        self.driver = RcodeZeroDNSDriver('mysecret')
        self.test_zone = Zone(id='example.at', domain='example.at', driver=self.driver, type='master', ttl=None, extra={})
        self.test_record = Record(id=None, name='', data='192.0.2.1', type=RecordType.A, zone=self.test_zone, driver=self.driver, extra={})

    def test_create_record(self):
        record = self.test_zone.create_record(name='newrecord', type=RecordType.A, data='192.0.5.4', extra={'ttl': 86400})
        self.assertEqual(record.id, None)
        self.assertEqual(record.name, 'newrecord')
        self.assertEqual(record.data, '192.0.5.4')
        self.assertEqual(record.type, RecordType.A)
        self.assertEqual(record.ttl, 86400)

    def test_create_zone(self):
        extra = {'masters': ['193.0.2.2']}
        zone = self.driver.create_zone('example.at', type='slave', extra=extra)
        self.assertEqual(zone.id, 'example.at.')
        self.assertEqual(zone.domain, 'example.at')
        self.assertEqual(zone.type, 'slave')
        self.assertEqual(zone.ttl, None)

    def test_delete_record(self):
        self.assertTrue(self.test_record.delete())

    def test_delete_zone(self):
        self.assertTrue(self.test_zone.delete())

    def test_get_record(self):
        record = self.driver.get_record('example.at.', '12345')
        self.assertEqual(record, None)

    def test_get_zone(self):
        zone = self.driver.get_zone('example.at')
        self.assertEqual(zone.id, 'example.at')
        self.assertEqual(zone.domain, 'example.at')
        self.assertEqual(zone.type, 'master')
        self.assertEqual(zone.ttl, None)

    def test_list_record_types(self):
        result = self.driver.list_record_types()
        self.assertEqual(len(result), 25)

    def test_list_records(self):
        records = self.driver.list_records(self.test_zone)
        self.assertEqual(len(records), 3)

    def test_list_zones(self):
        zones = self.driver.list_zones()
        self.assertEqual(zones[0].id, 'example1.at')
        self.assertEqual(zones[0].domain, 'example1.at')
        self.assertEqual(zones[0].type, 'slave')
        self.assertEqual(zones[0].ttl, None)
        self.assertEqual(zones[0].extra['masters'][0], '193.0.2.2')
        self.assertEqual(zones[0].extra['serial'], 20180411)
        self.assertEqual(zones[1].id, 'example2.at')
        self.assertEqual(zones[1].domain, 'example2.at')
        self.assertEqual(zones[1].type, 'master')
        self.assertEqual(zones[1].ttl, None)

    def test_update_record(self):
        record = self.driver.update_record(self.test_record, name='newrecord', type=RecordType.A, data='127.0.0.1', extra={'ttl': 300})
        self.assertEqual(record.id, '430059068a39501b8df5a577b1e46ad7')
        self.assertEqual(record.name, 'newrecord')
        self.assertEqual(record.data, '127.0.0.1')
        self.assertEqual(record.type, RecordType.A)
        self.assertEqual(record.ttl, 300)

    def test_update_zone(self):
        with self.assertRaises(NotImplementedError):
            self.driver.update_zone(self.test_zone, 'example.at')

    def test_create_existing_zone(self):
        RcodeZeroDNSMockHttp.type = 'EXISTS'
        extra = {'masters': ['193.0.2.2']}
        with self.assertRaises(ZoneAlreadyExistsError):
            self.driver.create_zone('example1.at', type='slave', extra=extra)

    def test_get_missing_zone(self):
        RcodeZeroDNSMockHttp.type = 'MISSING'
        with self.assertRaises(ZoneDoesNotExistError):
            self.driver.get_zone('example.com')

    def test_delete_missing_zone(self):
        RcodeZeroDNSMockHttp.type = 'MISSING'
        self.assertFalse(self.test_zone.delete())