import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_LUADNS
from libcloud.dns.drivers.luadns import LuadnsDNSDriver
from libcloud.test.file_fixtures import DNSFileFixtures
class LuadnsMockHttp(MockHttp):
    fixtures = DNSFileFixtures('luadns')

    def _v1_zones(self, method, url, body, headers):
        body = self.fixtures.load('zones_list.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_zones_EMPTY_ZONES_LIST(self, method, url, body, headers):
        body = self.fixtures.load('empty_zones_list.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_zones_13_ZONE_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('zone_does_not_exist.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_zones_31_GET_ZONE_SUCCESS(self, method, url, body, headers):
        body = self.fixtures.load('get_zone.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_zones_11_DELETE_ZONE_SUCCESS(self, method, url, body, headers):
        body = self.fixtures.load('delete_zone_success.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_zones_11_DELETE_ZONE_ZONE_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('zone_does_not_exist.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_zones_CREATE_ZONE_SUCCESS(self, method, url, body, headers):
        body = self.fixtures.load('create_zone_success.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_zones_CREATE_ZONE_ZONE_ALREADY_EXISTS(self, method, url, body, headers):
        body = self.fixtures.load('zone_already_exists.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_zones_11_records_EMPTY_RECORDS_LIST(self, method, url, body, headers):
        body = self.fixtures.load('empty_records_list.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_zones_11_records_LIST_RECORDS_SUCCESS(self, method, url, body, headers):
        body = self.fixtures.load('records_list.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_zones_31_records_31_GET_RECORD_RECORD_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('record_does_not_exist.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_zones_31_GET_RECORD_RECORD_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('get_zone.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_zones_31_GET_RECORD_SUCCESS(self, method, url, body, headers):
        body = self.fixtures.load('get_zone.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_zones_31_records_31_GET_RECORD_SUCCESS(self, method, url, body, headers):
        body = self.fixtures.load('get_record.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_zones_11_records_13_DELETE_RECORD_SUCCESS(self, method, url, body, headers):
        body = self.fixtures.load('delete_record_success.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_zones_11_records_13_DELETE_RECORD_RECORD_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('record_does_not_exist.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_zones_11_records_CREATE_RECORD_SUCCESS(self, method, url, body, headers):
        body = self.fixtures.load('create_record_success.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])