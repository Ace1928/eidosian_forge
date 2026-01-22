import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_DNSPOD
from libcloud.dns.drivers.dnspod import DNSPodDNSDriver
from libcloud.test.file_fixtures import DNSFileFixtures
def _Record_Remove_DELETE_RECORD_SUCCESS(self, method, url, body, headers):
    body = self.fixtures.load('delete_record_success.json')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])