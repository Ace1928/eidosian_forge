import sys
import json
from libcloud.test import MockHttp, LibcloudTestCase, unittest
from libcloud.dns.base import Zone
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.common.types import ProviderError
from libcloud.test.secrets import DNS_PARAMS_AURORADNS
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.auroradns import AuroraDNSDriver, AuroraDNSHealthCheckType
def test_res_to_record(self):
    res = {'id': 2, 'name': 'www', 'type': 'AAAA', 'content': '2001:db8:100', 'created': 1234, 'modified': 2345, 'disabled': False, 'ttl': 1800, 'prio': 10}
    zone = Zone(id=1, domain='example.com', type=None, ttl=60, driver=self.driver)
    record = self.driver._AuroraDNSDriver__res_to_record(zone, res)
    self.assertEqual(res['name'], record.name)
    self.assertEqual(res['ttl'], record.extra['ttl'])
    self.assertEqual(res['prio'], record.extra['priority'])
    self.assertEqual(res['type'], record.type)
    self.assertEqual(res['content'], record.data)
    self.assertEqual(zone, record.zone)
    self.assertEqual(self.driver, record.driver)