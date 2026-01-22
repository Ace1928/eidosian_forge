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
def _v1_0_11111_status_251c0d0c_95bc_4e09_b99f_4b8748b66246(self, method, url, body, headers):
    body = self.fixtures.load('update_record_success.json')
    return (httplib.OK, body, self.base_headers, httplib.responses[httplib.OK])