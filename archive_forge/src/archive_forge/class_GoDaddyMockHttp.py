import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.base import Zone, RecordType
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_GODADDY
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.godaddy import GoDaddyDNSDriver
class GoDaddyMockHttp(MockHttp):
    fixtures = DNSFileFixtures('godaddy')

    def _v1_domains(self, method, url, body, headers):
        body = self.fixtures.load('v1_domains.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_domains_aperture_platform_com(self, method, url, body, headers):
        body = self.fixtures.load('v1_domains_aperture_platform_com.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_domains_aperture_platform_com_records(self, method, url, body, headers):
        body = self.fixtures.load('v1_domains_aperture_platform_com_records.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_domains_available(self, method, url, body, headers):
        body = self.fixtures.load('v1_domains_available.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_domains_tlds(self, method, url, body, headers):
        body = self.fixtures.load('v1_domains_tlds.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_domains_aperture_platform_com_records_A_www(self, method, url, body, headers):
        body = self.fixtures.load('v1_domains_aperture_platform_com_records_A_www.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_domains_purchase_schema_com(self, method, url, body, headers):
        body = self.fixtures.load('v1_domains_purchase_schema_com.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_domains_agreements(self, method, url, body, headers):
        body = self.fixtures.load('v1_domains_agreements.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_domains_purchase(self, method, url, body, headers):
        body = self.fixtures.load('v1_domains_purchase.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])