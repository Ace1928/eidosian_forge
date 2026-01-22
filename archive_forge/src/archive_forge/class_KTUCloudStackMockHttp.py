import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib, urlparse, parse_qsl
from libcloud.test.compute import TestCaseMixin
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.ktucloud import KTUCloudNodeDriver
class KTUCloudStackMockHttp(MockHttp, unittest.TestCase):
    fixtures = ComputeFileFixtures('ktucloud')
    fixture_tag = 'default'

    def _load_fixture(self, fixture):
        body = self.fixtures.load(fixture)
        return (body, json.loads(body))

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

    def _cmd_queryAsyncJobResult(self, jobid):
        fixture = 'queryAsyncJobResult' + '_' + str(jobid) + '.json'
        body, obj = self._load_fixture(fixture)
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])