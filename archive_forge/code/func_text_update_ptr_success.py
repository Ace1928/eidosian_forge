import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.common.types import LibcloudError
from libcloud.compute.base import Node
from libcloud.test.secrets import DNS_PARAMS_RACKSPACE
from libcloud.loadbalancer.base import LoadBalancer
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.rackspace import RackspaceDNSDriver, RackspacePTRRecord
def text_update_ptr_success(self):
    records = self.driver.ex_iterate_ptr_records(RDNS_NODE)
    original = next(records)
    updated = self.driver.ex_update_ptr_record(original, domain=original.domain)
    self.assertEqual(original.id, updated.id)
    extra_update = {'ttl': original.extra['ttl']}
    updated = self.driver.ex_update_ptr_record(original, extra=extra_update)
    self.assertEqual(original.id, updated.id)
    updated = self.driver.ex_update_ptr_record(original, 'new-domain')
    self.assertEqual(original.id, updated.id)