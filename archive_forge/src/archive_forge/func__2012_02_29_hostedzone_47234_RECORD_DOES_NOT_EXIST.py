import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_ROUTE53
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.route53 import Route53DNSDriver
def _2012_02_29_hostedzone_47234_RECORD_DOES_NOT_EXIST(self, method, url, body, headers):
    body = self.fixtures.load('get_zone.xml')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])