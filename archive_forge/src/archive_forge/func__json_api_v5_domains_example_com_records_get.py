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
def _json_api_v5_domains_example_com_records_get(self, method, url, body, headers):
    body = self.fixtures.load('list_records.json')
    resp_headers = {}
    if headers is not None and 'Accept' in headers and (headers['Accept'] == 'text/plain'):
        body = self.fixtures.load('list_records_bind.txt')
        resp_headers['Content-Type'] = 'text/plain'
    return (httplib.OK, body, resp_headers, httplib.responses[httplib.OK])