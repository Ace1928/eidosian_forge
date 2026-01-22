import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import VULTR_PARAMS
from libcloud.dns.drivers.vultr import VultrDNSDriver, VultrDNSDriverV1
from libcloud.test.file_fixtures import DNSFileFixtures
class VultrTests(unittest.TestCase):

    def setUp(self):
        VultrMockHttp.type = None
        VultrDNSDriverV1.connectionCls.conn_class = VultrMockHttp
        self.driver = VultrDNSDriver(*VULTR_PARAMS, api_version='1')
        self.test_zone = Zone(id='test.com', type='master', ttl=None, domain='test.com', extra={}, driver=self)
        self.test_record = Record(id='31', type=RecordType.A, name='test', zone=self.test_zone, data='127.0.0.1', driver=self, extra={})

    def test_correct_class_is_used(self):
        self.assertIsInstance(self.driver, VultrDNSDriverV1)

    def test_list_zones_empty(self):
        VultrMockHttp.type = 'EMPTY_ZONES_LIST'
        zones = self.driver.list_zones()
        self.assertEqual(zones, [])

    def test_list_zones_success(self):
        zones = self.driver.list_zones()
        self.assertEqual(len(zones), 4)
        zone = zones[0]
        self.assertEqual(zone.id, 'example.com')
        self.assertEqual(zone.type, 'master')
        self.assertEqual(zone.domain, 'example.com')
        self.assertIsNone(zone.ttl)
        zone = zones[1]
        self.assertEqual(zone.id, 'zupo.com')
        self.assertEqual(zone.type, 'master')
        self.assertEqual(zone.domain, 'zupo.com')
        self.assertIsNone(zone.ttl)
        zone = zones[2]
        self.assertEqual(zone.id, 'oltjano.com')
        self.assertEqual(zone.type, 'master')
        self.assertEqual(zone.domain, 'oltjano.com')
        self.assertIsNone(zone.ttl)
        zone = zones[3]
        self.assertEqual(zone.id, '13.com')
        self.assertEqual(zone.type, 'master')
        self.assertEqual(zone.domain, '13.com')
        self.assertIsNone(zone.ttl)

    def test_get_zone_zone_does_not_exist(self):
        VultrMockHttp.type = 'GET_ZONE_ZONE_DOES_NOT_EXIST'
        try:
            self.driver.get_zone(zone_id='test.com')
        except ZoneDoesNotExistError as e:
            self.assertEqual(e.zone_id, 'test.com')
        else:
            self.fail('Exception was not thrown')

    def test_get_zone_success(self):
        VultrMockHttp.type = 'GET_ZONE_SUCCESS'
        zone = self.driver.get_zone(zone_id='zupo.com')
        self.assertEqual(zone.id, 'zupo.com')
        self.assertEqual(zone.domain, 'zupo.com')
        self.assertEqual(zone.type, 'master')
        self.assertIsNone(zone.ttl)

    def test_delete_zone_zone_does_not_exist(self):
        VultrMockHttp.type = 'DELETE_ZONE_ZONE_DOES_NOT_EXIST'
        try:
            self.driver.delete_zone(zone=self.test_zone)
        except ZoneDoesNotExistError as e:
            self.assertEqual(e.zone_id, self.test_zone.id)
        else:
            self.fail('Exception was not thrown')

    def test_delete_zone_success(self):
        zone = self.driver.list_zones()[0]
        status = self.driver.delete_zone(zone=zone)
        self.assertTrue(status)

    def test_create_zone_success(self):
        zone = self.driver.create_zone(domain='test.com', extra={'serverip': '127.0.0.1'})
        self.assertEqual(zone.id, 'test.com')
        self.assertEqual(zone.domain, 'test.com')
        (self.assertEqual(zone.type, 'master'),)
        self.assertIsNone(zone.ttl)

    def test_create_zone_zone_already_exists(self):
        VultrMockHttp.type = 'CREATE_ZONE_ZONE_ALREADY_EXISTS'
        try:
            self.driver.create_zone(domain='example.com', extra={'serverip': '127.0.0.1'})
        except ZoneAlreadyExistsError as e:
            self.assertEqual(e.zone_id, 'example.com')
        else:
            self.fail('Exception was not thrown')

    def test_get_record_record_does_not_exist(self):
        VultrMockHttp.type = 'GET_RECORD_RECORD_DOES_NOT_EXIST'
        try:
            self.driver.get_record(zone_id='zupo.com', record_id='1300')
        except RecordDoesNotExistError as e:
            self.assertEqual(e.record_id, '1300')
        else:
            self.fail('Exception was not thrown')

    def test_list_records_zone_does_not_exist(self):
        VultrMockHttp.type = 'LIST_RECORDS_ZONE_DOES_NOT_EXIST'
        try:
            self.driver.list_records(zone=self.test_zone)
        except ZoneDoesNotExistError as e:
            self.assertEqual(e.zone_id, self.test_zone.id)
        else:
            self.fail('Exception was not thrown')

    def test_list_records_empty(self):
        VultrMockHttp.type = 'EMPTY_RECORDS_LIST'
        zone = self.driver.list_zones()[0]
        records = self.driver.list_records(zone=zone)
        self.assertEqual(records, [])

    def test_list_records_success(self):
        zone = self.driver.get_zone(zone_id='zupo.com')
        records = self.driver.list_records(zone=zone)
        self.assertEqual(len(records), 2)
        arecord = records[0]
        self.assertEqual(arecord.id, '13')
        self.assertEqual(arecord.name, 'arecord')
        self.assertEqual(arecord.type, RecordType.A)
        self.assertEqual(arecord.data, '127.0.0.1')

    def test_get_record_success(self):
        VultrMockHttp.type = 'GET_RECORD'
        record = self.driver.get_record(zone_id='zupo.com', record_id='1300')
        self.assertEqual(record.id, '1300')
        self.assertEqual(record.name, 'zupo')
        self.assertEqual(record.data, '127.0.0.1')
        self.assertEqual(record.type, RecordType.A)

    def test_delete_record_record_does_not_exist(self):
        VultrMockHttp.type = 'DELETE_RECORD_RECORD_DOES_NOT_EXIST'
        try:
            self.driver.delete_record(record=self.test_record)
        except RecordDoesNotExistError as e:
            self.assertEqual(e.record_id, self.test_record.id)
        else:
            self.fail('Exception was not thrown')

    def test_delete_record_success(self):
        zone = self.driver.list_zones()[0]
        record = self.driver.list_records(zone=zone)[0]
        status = self.driver.delete_record(record=record)
        self.assertTrue(status)