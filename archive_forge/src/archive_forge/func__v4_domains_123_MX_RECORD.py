import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import RecordType
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_LINODE
from libcloud.dns.drivers.linode import LinodeDNSDriver, LinodeDNSDriverV4
from libcloud.test.file_fixtures import DNSFileFixtures
def _v4_domains_123_MX_RECORD(self, method, url, body, headers):
    body = self.fixtures.load('get_zone.json')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])