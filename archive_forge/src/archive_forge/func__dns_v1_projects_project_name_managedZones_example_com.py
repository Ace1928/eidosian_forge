import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_GOOGLE, DNS_KEYWORD_PARAMS_GOOGLE
from libcloud.common.google import GoogleBaseAuthConnection
from libcloud.dns.drivers.google import GoogleDNSDriver
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.test.common.test_google import GoogleTestCase, GoogleAuthMockHttp
def _dns_v1_projects_project_name_managedZones_example_com(self, method, url, body, headers):
    if method == 'GET':
        body = self.fixtures.load('managed_zones_1.json')
    elif method == 'DELETE':
        body = None
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])