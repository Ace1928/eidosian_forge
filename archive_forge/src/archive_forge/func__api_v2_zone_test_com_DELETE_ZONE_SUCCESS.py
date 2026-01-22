import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.base import Zone
from libcloud.dns.types import ZoneDoesNotExistError, ZoneAlreadyExistsError
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_BUDDYNS
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.buddyns import BuddyNSDNSDriver
def _api_v2_zone_test_com_DELETE_ZONE_SUCCESS(self, method, url, body, headers):
    body = self.fixtures.load('delete_zone_success.json')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])