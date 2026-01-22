import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_DNSPOD
from libcloud.dns.drivers.dnspod import DNSPodDNSDriver
from libcloud.test.file_fixtures import DNSFileFixtures
def _Domain_List_EMPTY_ZONES_LIST(self, method, url, body, headers):
    body = self.fixtures.load('empty_zones_list.json')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])