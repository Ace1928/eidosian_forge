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
class GoogleDNSMockHttp(MockHttp):
    fixtures = DNSFileFixtures('google')

    def _dns_v1_projects_project_name_managedZones(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('zone_create.json')
        else:
            body = self.fixtures.load('zone_list.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _dns_v1_projects_project_name_managedZones_FILTER_ZONES(self, method, url, body, headers):
        body = self.fixtures.load('zone_list.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _dns_v1_projects_project_name_managedZones_example_com_rrsets_FILTER_ZONES(self, method, url, body, headers):
        body = self.fixtures.load('record.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _dns_v1_projects_project_name_managedZones_example_com_rrsets(self, method, url, body, headers):
        body = self.fixtures.load('records_list.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _dns_v1_projects_project_name_managedZones_example_com(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('managed_zones_1.json')
        elif method == 'DELETE':
            body = None
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _dns_v1_projects_project_name_managedZones_example_com_changes(self, method, url, body, headers):
        body = self.fixtures.load('record_changes.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _dns_v1_projects_project_name_managedZones_example_com_ZONE_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('get_zone_does_not_exists.json')
        return (httplib.NOT_FOUND, body, {}, httplib.responses[httplib.NOT_FOUND])

    def _dns_v1_projects_project_name_managedZones_example_com_RECORD_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('managed_zones_1.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _dns_v1_projects_project_name_managedZones_example_com_rrsets_RECORD_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('no_record.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _dns_v1_projects_project_name_managedZones_example_com_rrsets_ZONE_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('get_zone_does_not_exists.json')
        return (httplib.NOT_FOUND, body, {}, httplib.responses[httplib.NOT_FOUND])

    def _dns_v1_projects_project_name_managedZones_example_com_FILTER_ZONES(self, method, url, body, headers):
        body = self.fixtures.load('zone.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])