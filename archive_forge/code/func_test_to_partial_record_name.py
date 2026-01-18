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
def test_to_partial_record_name(self):
    domain = 'example.com'
    names = ['test.example.com', 'foo.bar.example.com', 'example.com.example.com', 'example.com']
    expected_values = ['test', 'foo.bar', 'example.com', None]
    for name, expected_value in zip(names, expected_values):
        value = self.driver._to_partial_record_name(domain=domain, name=name)
        self.assertEqual(value, expected_value)