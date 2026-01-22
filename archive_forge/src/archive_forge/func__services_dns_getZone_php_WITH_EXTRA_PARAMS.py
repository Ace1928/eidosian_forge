import sys
import unittest
from unittest.mock import MagicMock
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_DURABLEDNS
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.durabledns import (
def _services_dns_getZone_php_WITH_EXTRA_PARAMS(self, method, url, body, headers):
    body = self.fixtures.load('get_zone_WITH_EXTRA_PARAMS.xml')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])