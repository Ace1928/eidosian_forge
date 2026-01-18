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
def test_update_multivalue_record(self):
    record = self.driver.get_record(self.test_zone.id, 'MX:lists')
    updated = self.driver.update_record(record, None, None, 'mail1', {'ttl': 400, 'priority': 10})
    self.assertEqual(updated.extra['priority'], '10')
    self.assertEqual(updated.data, 'mail1')
    self.assertTrue('_other_records' in record.extra)
    other_record = record.extra['_other_records'][0]
    self.assertEqual(other_record['extra']['priority'], '20')