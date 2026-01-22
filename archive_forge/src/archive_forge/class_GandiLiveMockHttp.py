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
class GandiLiveMockHttp(BaseGandiLiveMockHttp):
    fixtures = DNSFileFixtures('gandi_live')

    def _json_api_v5_domains_get(self, method, url, body, headers):
        body = self.fixtures.load('list_zones.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _json_api_v5_domains_example_com_get(self, method, url, body, headers):
        body = self.fixtures.load('get_zone.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _json_api_v5_domains_badexample_com_get(self, method, url, body, headers):
        body = self.fixtures.load('get_bad_zone.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _json_api_v5_domains_nosuchzone_com_get(self, method, url, body, headers):
        body = self.fixtures.load('get_nonexistent_zone.json')
        return (httplib.NOT_FOUND, body, {}, httplib.responses[httplib.NOT_FOUND])

    def _json_api_v5_zones_post(self, method, url, body, headers):
        input = json.loads(body)
        if 'badexample' in input['name']:
            body = self.fixtures.load('create_bad_zone.json')
            return (httplib.CONFLICT, body, {}, httplib.responses[httplib.CONFLICT])
        else:
            body = self.fixtures.load('create_zone.json')
            return (httplib.OK, body, {'Location': '/zones/54321'}, httplib.responses[httplib.OK])

    def _json_api_v5_zones_111111_patch(self, method, url, body, headers):
        body = self.fixtures.load('update_gandi_zone.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _json_api_v5_zones_111111_delete(self, method, url, body, headers):
        return (httplib.NO_CONTENT, '', {}, httplib.responses[httplib.OK])

    def _json_api_v5_domains_example_org_patch(self, method, url, body, headers):
        body = self.fixtures.load('create_domain.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _json_api_v5_domains_example_com_records_get(self, method, url, body, headers):
        body = self.fixtures.load('list_records.json')
        resp_headers = {}
        if headers is not None and 'Accept' in headers and (headers['Accept'] == 'text/plain'):
            body = self.fixtures.load('list_records_bind.txt')
            resp_headers['Content-Type'] = 'text/plain'
        return (httplib.OK, body, resp_headers, httplib.responses[httplib.OK])

    def _json_api_v5_domains_example_com_records_bob_A_get(self, method, url, body, headers):
        body = self.fixtures.load('get_record.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _json_api_v5_domains_example_com_records_none_A_get(self, method, url, body, headers):
        body = self.fixtures.load('get_nonexistent_record.json')
        return (httplib.NOT_FOUND, body, {}, httplib.responses[httplib.NOT_FOUND])

    def _json_api_v5_domains_example_com_records_lists_MX_get(self, method, url, body, headers):
        body = self.fixtures.load('get_mx_record.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _json_api_v5_domains_example_com_records_lists_MX_put(self, method, url, body, headers):
        body = self.fixtures.load('update_mx_record.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _json_api_v5_domains_example_com_records_post(self, method, url, body, headers):
        body = self.fixtures.load('create_record.json')
        return (httplib.OK, body, {'Location': '/zones/12345/records/alice/AAAA'}, httplib.responses[httplib.OK])

    def _json_api_v5_domains_badexample_com_records_post(self, method, url, body, headers):
        body = self.fixtures.load('create_existing_record.json')
        return (httplib.CONFLICT, body, {}, httplib.responses[httplib.CONFLICT])

    def _json_api_v5_domains_badexample_com_records_get(self, method, url, body, headers):
        return (httplib.INTERNAL_SERVER_ERROR, body, {}, httplib.responses[httplib.INTERNAL_SERVER_ERROR])

    def _json_api_v5_domains_badexample_com_records_jane_A_put(self, method, url, body, headers):
        body = self.fixtures.load('update_bad_record.json')
        return (httplib.BAD_REQUEST, body, {}, httplib.responses[httplib.BAD_REQUEST])

    def _json_api_v5_domains_example_com_records_bob_A_put(self, method, url, body, headers):
        body = self.fixtures.load('update_record.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _json_api_v5_domains_example_com_records_bob_A_delete(self, method, url, body, headers):
        return (httplib.NO_CONTENT, '', {}, httplib.responses[httplib.OK])