import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlparse, parse_qsl, assertRaisesRegex
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import Provider
from libcloud.test.file_fixtures import LoadBalancerFileFixtures
from libcloud.loadbalancer.providers import get_driver
from libcloud.loadbalancer.drivers.cloudstack import CloudStackLBDriver
def _test_path(self, method, url, body, headers):
    url = urlparse.urlparse(url)
    query = dict(parse_qsl(url.query))
    self.assertTrue('apiKey' in query)
    self.assertTrue('command' in query)
    self.assertTrue('response' in query)
    self.assertTrue('signature' in query)
    self.assertTrue(query['response'] == 'json')
    del query['apiKey']
    del query['response']
    del query['signature']
    command = query.pop('command')
    if hasattr(self, '_cmd_' + command):
        return getattr(self, '_cmd_' + command)(**query)
    else:
        fixture = command + '_' + self.fixture_tag + '.json'
        body, obj = self._load_fixture(fixture)
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])