import sys
import unittest
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.dns.types import RecordType
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DIGITALOCEAN_v2_PARAMS
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.digitalocean import DigitalOceanDNSDriver
class DigitalOceanDNSMockHttp(MockHttp):
    fixtures = DNSFileFixtures('digitalocean')
    response_map = {None: httplib.OK, 'CREATE': httplib.CREATED, 'DELETE': httplib.NO_CONTENT, 'EMPTY': httplib.OK, 'NOT_FOUND': httplib.NOT_FOUND, 'UNAUTHORIZED': httplib.UNAUTHORIZED, 'UPDATE': httplib.OK, 'UNPROCESSABLE': httplib.UNPROCESSABLE_ENTITY}

    def _v2_domains(self, method, url, body, headers):
        body = self.fixtures.load('_v2_domains.json')
        return (self.response_map[self.type], body, {}, httplib.responses[self.response_map[self.type]])

    def _v2_domains_CREATE(self, method, url, body, headers):
        if body is None:
            body = self.fixtures.load('_v2_domains_UNPROCESSABLE_ENTITY.json')
            return (self.response_map['UNPROCESSABLE'], body, {}, httplib.responses[self.response_map['UNPROCESSABLE']])
        body = self.fixtures.load('_v2_domains_CREATE.json')
        return (self.response_map[self.type], body, {}, httplib.responses[self.response_map[self.type]])

    def _v2_domains_testdomain(self, method, url, body, headers):
        body = self.fixtures.load('_v2_domains_testdomain.json')
        return (self.response_map[self.type], body, {}, httplib.responses[self.response_map[self.type]])

    def _v2_domains_testdomain_DELETE(self, method, url, body, headers):
        return (self.response_map[self.type], body, {}, httplib.responses[self.response_map[self.type]])

    def _v2_domains_testdomain_NOT_FOUND(self, method, url, body, headers):
        body = self.fixtures.load('_v2_domains_testdomain_NOT_FOUND.json')
        return (self.response_map[self.type], body, {}, httplib.responses[self.response_map[self.type]])

    def _v2_domains_testdomain_records(self, method, url, body, headers):
        body = self.fixtures.load('_v2_domains_testdomain_records.json')
        return (self.response_map[self.type], body, {}, httplib.responses[self.response_map[self.type]])

    def _v2_domains_testdomain_records_CREATE(self, method, url, body, headers):
        if body is None:
            body = self.fixtures.load('_v2_domains_UNPROCESSABLE_ENTITY.json')
            return (self.response_map['UNPROCESSABLE'], body, {}, httplib.responses[self.response_map['UNPROCESSABLE']])
        body = self.fixtures.load('_v2_domains_testdomain_records_CREATE.json')
        return (self.response_map[self.type], body, {}, httplib.responses[self.response_map[self.type]])

    def _v2_domains_testdomain_records_1234564(self, method, url, body, headers):
        body = self.fixtures.load('_v2_domains_testdomain_records_1234564.json')
        return (self.response_map[self.type], body, {}, httplib.responses[self.response_map[self.type]])

    def _v2_domains_testdomain_records_1234564_DELETE(self, method, url, body, headers):
        self.type = 'DELETE'
        return (self.response_map[self.type], body, {}, httplib.responses[self.response_map[self.type]])

    def _v2_domains_testdomain_records_1234564_UPDATE(self, method, url, body, headers):
        if body is None:
            body = self.fixtures.load('_v2_domains_UNPROCESSABLE_ENTITY.json')
            return (self.response_map['UNPROCESSABLE'], body, {}, httplib.responses[self.response_map['UNPROCESSABLE']])
        body = self.fixtures.load('_v2_domains_testdomain_records_1234564_UPDATE.json')
        return (self.response_map[self.type], body, {}, httplib.responses[self.response_map[self.type]])