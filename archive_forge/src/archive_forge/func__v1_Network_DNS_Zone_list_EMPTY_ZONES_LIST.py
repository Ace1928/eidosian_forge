import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_LIQUIDWEB
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.liquidweb import LiquidWebDNSDriver
def _v1_Network_DNS_Zone_list_EMPTY_ZONES_LIST(self, method, url, body, headers):
    body = self.fixtures.load('empty_zones_list.json')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])