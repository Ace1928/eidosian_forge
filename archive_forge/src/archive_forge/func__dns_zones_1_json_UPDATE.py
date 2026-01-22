import sys
import unittest
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.dns.types import RecordType
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_ONAPP
from libcloud.common.exceptions import BaseHTTPError
from libcloud.dns.drivers.onapp import OnAppDNSDriver
from libcloud.test.file_fixtures import DNSFileFixtures
def _dns_zones_1_json_UPDATE(self, method, url, body, headers):
    body = self.fixtures.load('get_zone.json')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])