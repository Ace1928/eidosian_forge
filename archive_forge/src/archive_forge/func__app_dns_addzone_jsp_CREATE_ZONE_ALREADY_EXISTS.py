import sys
import unittest
from unittest.mock import MagicMock
from libcloud.test import MockHttp
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_ZONOMI
from libcloud.dns.drivers.zonomi import ZonomiDNSDriver
from libcloud.test.file_fixtures import DNSFileFixtures
def _app_dns_addzone_jsp_CREATE_ZONE_ALREADY_EXISTS(self, method, url, body, headers):
    body = self.fixtures.load('create_zone_already_exists.xml')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])