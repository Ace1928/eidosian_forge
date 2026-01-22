import sys
import json
from libcloud.test import MockHttp, unittest
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib, urlparse
from libcloud.common.types import LibcloudError
from libcloud.test.secrets import DNS_PARAMS_CLOUDFLARE
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.cloudflare import (
class CloudFlareMockHttp(MockHttp, unittest.TestCase):
    fixtures = DNSFileFixtures('cloudflare')

    def _client_v4_memberships(self, method, url, body, headers):
        if method not in {'GET'}:
            raise AssertionError('Unsupported method')
        body = self.fixtures.load('memberships_{}.json'.format(method))
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _client_v4_zones(self, method, url, body, headers):
        if method not in {'GET', 'POST'}:
            raise AssertionError('Unsupported method')
        body = self.fixtures.load('zones_{}.json'.format(method))
        return (httplib.BAD_REQUEST, body, {}, httplib.responses[httplib.BAD_REQUEST])

    def _client_v4_zones_1234(self, method, url, body, headers):
        if method not in {'GET', 'PATCH', 'DELETE'}:
            raise AssertionError('Unsupported method')
        body = self.fixtures.load('zone_{}.json'.format(method))
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _client_v4_zones_0000(self, method, url, body, headers):
        if method not in {'GET'}:
            raise AssertionError('Unsupported method')
        body = self.fixtures.load('zone_{}_404.json'.format(method))
        return (httplib.NOT_FOUND, body, {}, httplib.responses[httplib.NOT_FOUND])

    def _client_v4_zones_invalid(self, method, url, body, headers):
        if method not in {'GET'}:
            raise AssertionError('Unsupported method')
        body = self.fixtures.load('zone_{}_400.json'.format(method))
        return (httplib.BAD_REQUEST, body, {}, httplib.responses[httplib.BAD_REQUEST])

    def _client_v4_zones_1234_dns_records(self, method, url, body, headers):
        if method not in {'GET', 'POST'}:
            raise AssertionError('Unsupported method')
        url = urlparse.urlparse(url)
        if method == 'GET' and url.query:
            query = urlparse.parse_qs(url.query)
            page = query['page'][0]
            body = self.fixtures.load('records_{}_{}.json'.format(method, page))
        else:
            body = self.fixtures.load('records_{}.json'.format(method))
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _client_v4_zones_1234_dns_records_caa_record_type(self, method, url, body, headers):
        if method not in ['POST']:
            raise AssertionError('Unsupported method: %s' % method)
        url = urlparse.urlparse(url)
        body = json.loads(body)
        self.assertEqual(body['content'], '0\tissue\tcaa.example.com')
        body = self.fixtures.load('records_{}.json'.format(method))
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _client_v4_zones_1234_dns_records_sshfp_record_type(self, method, url, body, headers):
        if method not in ['POST']:
            raise AssertionError('Unsupported method: %s' % method)
        url = urlparse.urlparse(url)
        body = json.loads(body)
        expected_data = {'algorithm': '2', 'type': '1', 'fingerprint': 'ABCDEF12345'}
        self.assertEqual(body['data'], expected_data)
        body = self.fixtures.load('records_{}_sshfp.json'.format(method))
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _client_v4_zones_1234_dns_records_error_chain_error(self, method, url, body, headers):
        if method not in ['POST']:
            raise AssertionError('Unsupported method: %s' % method)
        body = self.fixtures.load('error_with_error_chain.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _client_v4_zones_1234_dns_records_0000(self, method, url, body, headers):
        if method not in {'GET'}:
            raise AssertionError('Unsupported method')
        body = self.fixtures.load('record_{}_404.json'.format(method))
        return (httplib.NOT_FOUND, body, {}, httplib.responses[httplib.NOT_FOUND])

    def _client_v4_zones_1234_dns_records_invalid(self, method, url, body, headers):
        if method not in {'GET'}:
            raise AssertionError('Unsupported method')
        body = self.fixtures.load('record_{}_400.json'.format(method))
        return (httplib.BAD_REQUEST, body, {}, httplib.responses[httplib.BAD_REQUEST])

    def _client_v4_zones_1234_dns_records_364797364(self, method, url, body, headers):
        if method not in {'GET', 'PUT', 'DELETE'}:
            raise AssertionError('Unsupported method')
        body = self.fixtures.load('record_{}.json'.format(method))
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])