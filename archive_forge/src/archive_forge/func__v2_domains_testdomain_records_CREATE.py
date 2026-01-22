import sys
import unittest
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.dns.types import RecordType
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DIGITALOCEAN_v2_PARAMS
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.digitalocean import DigitalOceanDNSDriver
def _v2_domains_testdomain_records_CREATE(self, method, url, body, headers):
    if body is None:
        body = self.fixtures.load('_v2_domains_UNPROCESSABLE_ENTITY.json')
        return (self.response_map['UNPROCESSABLE'], body, {}, httplib.responses[self.response_map['UNPROCESSABLE']])
    body = self.fixtures.load('_v2_domains_testdomain_records_CREATE.json')
    return (self.response_map[self.type], body, {}, httplib.responses[self.response_map[self.type]])