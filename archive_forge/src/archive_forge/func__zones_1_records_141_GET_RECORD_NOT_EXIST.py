import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_POINTDNS
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.pointdns import PointDNSDriver, PointDNSException
def _zones_1_records_141_GET_RECORD_NOT_EXIST(self, method, url, body, headers):
    body = self.fixtures.load('not_found.json')
    return (httplib.NOT_FOUND, body, {}, httplib.responses[httplib.NOT_FOUND])