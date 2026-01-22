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
def _zones_ffb62570_8414_4578_a346_526b44e320b7_records_5592f1ff(self, method, url, body, headers):
    body = None
    if method == 'GET':
        body = self.fixtures.load('zone_example_com_record_localhost.json')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])