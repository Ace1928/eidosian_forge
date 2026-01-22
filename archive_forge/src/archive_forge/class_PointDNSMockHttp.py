import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_POINTDNS
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.pointdns import PointDNSDriver, PointDNSException
class PointDNSMockHttp(MockHttp):
    fixtures = DNSFileFixtures('pointdns')

    def _zones_GET(self, method, url, body, headers):
        body = self.fixtures.load('_zones_GET.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _zones_CREATE(self, method, url, body, headers):
        body = self.fixtures.load('_zones_CREATE.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _zones_CREATE_ZONE_WITH_ERROR(self, method, url, body, headers):
        body = self.fixtures.load('error.json')
        return (httplib.PAYMENT_REQUIRED, body, {}, httplib.responses[httplib.PAYMENT_REQUIRED])

    def _zones_1_GET(self, method, url, body, headers):
        body = self.fixtures.load('_zones_GET_1.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _zones_1_ZONE_UPDATE(self, method, url, body, headers):
        body = self.fixtures.load('_zones_1_ZONE_UPDATE.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _zones_1_UPDATE_ZONE_WITH_ERROR(self, method, url, body, headers):
        body = self.fixtures.load('error.json')
        return (httplib.PAYMENT_REQUIRED, body, {}, httplib.responses[httplib.PAYMENT_REQUIRED])

    def _zones_1_GET_ZONE_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('not_found.json')
        return (httplib.NOT_FOUND, body, {}, httplib.responses[httplib.NOT_FOUND])

    def _zones_example_com_UPDATE(self, method, url, body, headers):
        body = self.fixtures.load('_zones_example_com_UPDATE.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _zones_1_DELETE(self, method, url, body, headers):
        body = self.fixtures.load('_zones_1_DELETE.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _zones_1_DELETE_ZONE_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('not_found.json')
        return (httplib.NOT_FOUND, body, {}, httplib.responses[httplib.NOT_FOUND])

    def _zones_1_records_CREATE(self, method, url, body, headers):
        body = self.fixtures.load('_zones_example_com_records_CREATE.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _zones_1_records_CREATE_WITH_ERROR(self, method, url, body, headers):
        body = self.fixtures.load('error.json')
        return (httplib.PAYMENT_REQUIRED, body, {}, httplib.responses[httplib.PAYMENT_REQUIRED])

    def _zones_1_records_GET(self, method, url, body, headers):
        body = self.fixtures.load('_zones_1_records_GET.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _zones_1_records_141_GET_RECORD_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('not_found.json')
        return (httplib.NOT_FOUND, body, {}, httplib.responses[httplib.NOT_FOUND])

    def _zones_1_records_141_GET(self, method, url, body, headers):
        body = self.fixtures.load('_zones_1_records_141_GET.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _zones_1_records_141_UPDATE(self, method, url, body, headers):
        body = self.fixtures.load('_zones_1_records_141_UPDATE.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _zones_1_records_141_UPDATE_RECORD_WITH_ERROR(self, method, url, body, headers):
        body = self.fixtures.load('error.json')
        return (httplib.PAYMENT_REQUIRED, body, {}, httplib.responses[httplib.PAYMENT_REQUIRED])

    def _zones_1_records_150_DELETE(self, method, url, body, headers):
        body = self.fixtures.load('_zones_1_records_150_DELETE.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _zones_1_records_150_DELETE_RECORD_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('not_found.json')
        return (httplib.NOT_FOUND, body, {}, httplib.responses[httplib.NOT_FOUND])

    def _zones_1_redirects_LIST(self, method, url, body, headers):
        body = self.fixtures.load('_zones_1_redirects_LIST.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _zones_1_mail_redirects_LIST(self, method, url, body, headers):
        body = self.fixtures.load('_zones_1_mail_redirects_LIST.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _zones_1_redirects_CREATE(self, method, url, body, headers):
        body = self.fixtures.load('_zones_1_redirects_CREATE.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _zones_1_redirects_CREATE_WITH_ERROR(self, method, url, body, headers):
        body = self.fixtures.load('redirect_error.json')
        return (httplib.METHOD_NOT_ALLOWED, body, {}, httplib.responses[httplib.METHOD_NOT_ALLOWED])

    def _zones_1_mail_redirects_CREATE(self, method, url, body, headers):
        body = self.fixtures.load('_zones_1_mail_redirects_CREATE.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _zones_1_mail_redirects_CREATE_WITH_ERROR(self, method, url, body, headers):
        body = self.fixtures.load('redirect_error.json')
        return (httplib.METHOD_NOT_ALLOWED, body, {}, httplib.responses[httplib.METHOD_NOT_ALLOWED])

    def _zones_1_redirects_36843229_GET_WITH_ERROR(self, method, url, body, headers):
        body = self.fixtures.load('redirect_error.json')
        return (httplib.METHOD_NOT_ALLOWED, body, {}, httplib.responses[httplib.METHOD_NOT_ALLOWED])

    def _zones_1_redirects_36843229_GET(self, method, url, body, headers):
        body = self.fixtures.load('_zones_1_redirects_GET.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _zones_1_redirects_36843229_GET_NOT_FOUND(self, method, url, body, headers):
        body = self.fixtures.load('not_found.json')
        return (httplib.NOT_FOUND, body, {}, httplib.responses[httplib.NOT_FOUND])

    def _zones_1_mail_redirects_5_GET(self, method, url, body, headers):
        body = self.fixtures.load('_zones_1_mail_redirects_GET.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _zones_1_mail_redirects_5_GET_WITH_ERROR(self, method, url, body, headers):
        body = self.fixtures.load('redirect_error.json')
        return (httplib.METHOD_NOT_ALLOWED, body, {}, httplib.responses[httplib.METHOD_NOT_ALLOWED])

    def _zones_1_redirects_36843229_UPDATE(self, method, url, body, headers):
        body = self.fixtures.load('_zones_1_redirects_UPDATE.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _zones_1_redirects_36843229_UPDATE_WITH_ERROR(self, method, url, body, headers):
        body = self.fixtures.load('redirect_error.json')
        return (httplib.METHOD_NOT_ALLOWED, body, {}, httplib.responses[httplib.METHOD_NOT_ALLOWED])

    def _zones_1_mail_redirects_5_UPDATE(self, method, url, body, headers):
        body = self.fixtures.load('_zones_1_mail_redirects_UPDATE.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _zones_1_mail_redirects_5_UPDATE_WITH_ERROR(self, method, url, body, headers):
        body = self.fixtures.load('redirect_error.json')
        return (httplib.METHOD_NOT_ALLOWED, body, {}, httplib.responses[httplib.METHOD_NOT_ALLOWED])

    def _zones_1_redirects_36843229_DELETE(self, method, url, body, headers):
        body = self.fixtures.load('_zones_1_redirects_DELETE.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _zones_1_mail_redirects_5_DELETE(self, method, url, body, headers):
        body = self.fixtures.load('_zones_1_mail_redirects_DELETE.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _zones_1_redirects_36843229_DELETE_WITH_ERROR(self, method, url, body, headers):
        body = self.fixtures.load('redirect_error.json')
        return (httplib.METHOD_NOT_ALLOWED, body, {}, httplib.responses[httplib.METHOD_NOT_ALLOWED])

    def _zones_1_mail_redirects_5_DELETE_WITH_ERROR(self, method, url, body, headers):
        body = self.fixtures.load('redirect_error.json')
        return (httplib.METHOD_NOT_ALLOWED, body, {}, httplib.responses[httplib.METHOD_NOT_ALLOWED])

    def _zones_1_redirects_36843229_DELETE_NOT_FOUND(self, method, url, body, headers):
        body = self.fixtures.load('not_found.json')
        return (httplib.NOT_FOUND, body, {}, httplib.responses[httplib.NOT_FOUND])

    def _zones_1_mail_redirects_5_DELETE_NOT_FOUND(self, method, url, body, headers):
        body = self.fixtures.load('not_found.json')
        return (httplib.NOT_FOUND, body, {}, httplib.responses[httplib.NOT_FOUND])