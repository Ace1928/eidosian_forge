import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.dns.drivers.vultr import VultrDNSDriver, VultrDNSDriverV2
from libcloud.test.file_fixtures import DNSFileFixtures
def _v2_domains_example_com_records_123(self, method, url, body, headers):
    if method == 'GET':
        body = self.fixtures.load('get_record.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])
    elif method == 'DELETE':
        return (httplib.NO_CONTENT, body, {}, httplib.responses[httplib.NO_CONTENT])
    elif method == 'PATCH':
        return (httplib.NO_CONTENT, body, {}, httplib.responses[httplib.NO_CONTENT])