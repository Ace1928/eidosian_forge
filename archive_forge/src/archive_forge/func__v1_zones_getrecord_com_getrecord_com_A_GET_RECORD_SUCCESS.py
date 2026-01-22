import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.common.nsone import NsOneException
from libcloud.test.secrets import DNS_PARAMS_NSONE
from libcloud.dns.drivers.nsone import NsOneDNSDriver
from libcloud.test.file_fixtures import DNSFileFixtures
def _v1_zones_getrecord_com_getrecord_com_A_GET_RECORD_SUCCESS(self, method, url, body, headers):
    body = self.fixtures.load('get_record_success.json')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])