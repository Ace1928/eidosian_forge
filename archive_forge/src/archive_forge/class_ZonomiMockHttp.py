import sys
import unittest
from unittest.mock import MagicMock
from libcloud.test import MockHttp
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_ZONOMI
from libcloud.dns.drivers.zonomi import ZonomiDNSDriver
from libcloud.test.file_fixtures import DNSFileFixtures
class ZonomiMockHttp(MockHttp):
    fixtures = DNSFileFixtures('zonomi')

    def _app_dns_dyndns_jsp_EMPTY_ZONES_LIST(self, method, url, body, headers):
        body = self.fixtures.load('empty_zones_list.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _app_dns_dyndns_jsp(self, method, url, body, headers):
        body = self.fixtures.load('list_zones.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _app_dns_dyndns_jsp_GET_ZONE_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('list_zones.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _app_dns_dyndns_jsp_GET_ZONE_SUCCESS(self, method, url, body, headers):
        body = self.fixtures.load('list_zones.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _app_dns_dyndns_jsp_DELETE_ZONE_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('delete_zone_does_not_exist.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _app_dns_dyndns_jsp_DELETE_ZONE_SUCCESS(self, method, url, body, headers):
        body = self.fixtures.load('delete_zone.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _app_dns_addzone_jsp_CREATE_ZONE_SUCCESS(self, method, url, body, headers):
        body = self.fixtures.load('create_zone.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _app_dns_addzone_jsp_CREATE_ZONE_ALREADY_EXISTS(self, method, url, body, headers):
        body = self.fixtures.load('create_zone_already_exists.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _app_dns_dyndns_jsp_LIST_RECORDS_EMPTY_LIST(self, method, url, body, headers):
        body = self.fixtures.load('list_records_empty_list.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _app_dns_dyndns_jsp_LIST_RECORDS_SUCCESS(self, method, url, body, headers):
        body = self.fixtures.load('list_records.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _app_dns_dyndns_jsp_DELETE_RECORD_SUCCESS(self, method, url, body, headers):
        body = self.fixtures.load('delete_record.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _app_dns_dyndns_jsp_DELETE_RECORD_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('delete_record_does_not_exist.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _app_dns_dyndns_jsp_CREATE_RECORD_SUCCESS(self, method, url, body, headers):
        body = self.fixtures.load('create_record.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _app_dns_dyndns_jsp_CREATE_RECORD_ALREADY_EXISTS(self, method, url, body, headers):
        body = self.fixtures.load('create_record_already_exists.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _app_dns_dyndns_jsp_GET_RECORD_SUCCESS(self, method, url, body, headers):
        body = self.fixtures.load('list_records.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _app_dns_dyndns_jsp_GET_RECORD_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('list_records.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _app_dns_converttosecondary_jsp(self, method, url, body, headers):
        body = self.fixtures.load('converted_to_slave.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _app_dns_converttosecondary_jsp_COULDNT_CONVERT(self, method, url, body, headers):
        body = self.fixtures.load('couldnt_convert.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _app_dns_converttomaster_jsp(self, method, url, body, headers):
        body = self.fixtures.load('converted_to_master.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _app_dns_converttomaster_jsp_COULDNT_CONVERT(self, method, url, body, headers):
        body = self.fixtures.load('couldnt_convert.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])