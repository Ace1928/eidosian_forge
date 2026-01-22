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
def _v1_0_11111_domains_CREATE_ZONE_VALIDATION_ERROR(self, method, url, body, headers):
    body = self.fixtures.load('create_zone_validation_error.json')
    return (httplib.BAD_REQUEST, body, self.base_headers, httplib.responses[httplib.BAD_REQUEST])