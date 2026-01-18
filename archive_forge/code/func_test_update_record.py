import sys
import json
from libcloud.test import MockHttp, unittest
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib, urlparse
from libcloud.common.types import LibcloudError
from libcloud.test.secrets import DNS_PARAMS_CLOUDFLARE
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.cloudflare import (
def test_update_record(self):
    zone = self.driver.list_zones()[0]
    record = zone.list_records()[0]
    updated_record = self.driver.update_record(record=record, name='test6', type=RecordType.A, data='127.0.0.4', extra={'proxied': True})
    self.assertEqual(updated_record.name, 'test6')
    self.assertEqual(updated_record.type, 'A')
    self.assertEqual(updated_record.data, '127.0.0.4')
    self.assertEqual(updated_record.extra['proxied'], True)