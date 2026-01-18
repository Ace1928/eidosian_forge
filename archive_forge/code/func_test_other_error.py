import sys
import json
import unittest
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_GANDI_LIVE
from libcloud.common.gandi_live import JsonParseError, GandiLiveBaseError, InvalidRequestError
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.gandi_live import GandiLiveDNSDriver
from libcloud.test.common.test_gandi_live import BaseGandiLiveMockHttp
def test_other_error(self):
    with self.assertRaises(GandiLiveBaseError):
        self.driver.list_records(self.test_bad_zone)