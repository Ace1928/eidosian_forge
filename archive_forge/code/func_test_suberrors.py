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
def test_suberrors(self):
    with self.assertRaises(InvalidRequestError) as ctx:
        self.driver.update_record(self.test_bad_record, 'jane', RecordType.A, '192.168.0.2', {'ttl': 500})
    self.assertTrue('is not a foo' in str(ctx.exception))