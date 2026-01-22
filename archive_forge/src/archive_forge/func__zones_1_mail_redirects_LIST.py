import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_POINTDNS
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.pointdns import PointDNSDriver, PointDNSException
def _zones_1_mail_redirects_LIST(self, method, url, body, headers):
    body = self.fixtures.load('_zones_1_mail_redirects_LIST.json')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])