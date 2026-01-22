import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import RecordType
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_LINODE
from libcloud.dns.drivers.linode import LinodeDNSDriver, LinodeDNSDriverV4
from libcloud.test.file_fixtures import DNSFileFixtures
class LinodeMockHttpV4(MockHttp):
    fixtures = DNSFileFixtures('linode_v4')

    def _v4_domains(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('list_zones.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        if method == 'POST':
            body = self.fixtures.load('create_zone.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v4_domains_123_records(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('list_records.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        if method == 'POST':
            body = self.fixtures.load('create_record.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v4_domains_123(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('get_zone.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        if method == 'PUT':
            body = self.fixtures.load('update_zone.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        if method == 'DELETE':
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v4_domains_123_ZONE_DOES_NOT_EXIST(self, method, url, body, headers):
        body = '{ "errors":[{"reason":"Not found"}]}'
        return (httplib.NOT_FOUND, body, {}, httplib.responses[httplib.NOT_FOUND])

    def _v4_domains_123_A_RECORD(self, method, url, body, headers):
        body = self.fixtures.load('get_zone.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v4_domains_123_MX_RECORD(self, method, url, body, headers):
        body = self.fixtures.load('get_zone.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v4_domains_123_records_123_A_RECORD(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('get_record_A.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        if method == 'PUT':
            body = self.fixtures.load('update_record.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v4_domains_123_records_123_MX_RECORD(self, method, url, body, headers):
        body = self.fixtures.load('get_record_MX.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v4_domains_123_records_123(self, method, url, body, headers):
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])