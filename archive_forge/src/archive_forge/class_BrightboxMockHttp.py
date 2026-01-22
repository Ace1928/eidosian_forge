import sys
import base64
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import b, httplib
from libcloud.common.types import InvalidCredsError
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import BRIGHTBOX_PARAMS
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.brightbox import BrightboxNodeDriver
class BrightboxMockHttp(MockHttp):
    fixtures = ComputeFileFixtures('brightbox')

    def _token(self, method, url, body, headers):
        if method == 'POST':
            return self.test_response(httplib.OK, self.fixtures.load('token.json'))

    def _token_INVALID_CLIENT(self, method, url, body, headers):
        if method == 'POST':
            return self.test_response(httplib.BAD_REQUEST, '{"error":"invalid_client"}')

    def _token_UNAUTHORIZED_CLIENT(self, method, url, body, headers):
        if method == 'POST':
            return self.test_response(httplib.UNAUTHORIZED, '{"error":"unauthorized_client"}')

    def _1_0_servers_INVALID_CLIENT(self, method, url, body, headers):
        return self.test_response(httplib.BAD_REQUEST, '{"error":"invalid_client"}')

    def _1_0_servers_UNAUTHORIZED_CLIENT(self, method, url, body, headers):
        return self.test_response(httplib.UNAUTHORIZED, '{"error":"unauthorized_client"}')

    def _1_0_images(self, method, url, body, headers):
        if method == 'GET':
            return self.test_response(httplib.OK, self.fixtures.load('list_images.json'))

    def _1_0_servers(self, method, url, body, headers):
        if method == 'GET':
            return self.test_response(httplib.OK, self.fixtures.load('list_servers.json'))
        elif method == 'POST':
            body = json.loads(body)
            encoded = base64.b64encode(b(USER_DATA)).decode('ascii')
            if 'user_data' in body and body['user_data'] != encoded:
                data = '{"error_name":"dodgy user data", "errors": ["User data not encoded properly"]}'
                return self.test_response(httplib.BAD_REQUEST, data)
            if body.get('zone', '') == 'zon-remk1':
                node = json.loads(self.fixtures.load('create_server_gb1_b.json'))
            else:
                node = json.loads(self.fixtures.load('create_server_gb1_a.json'))
            node['name'] = body['name']
            if 'server_groups' in body:
                node['server_groups'] = [{'id': x} for x in body['server_groups']]
            if 'user_data' in body:
                node['user_data'] = body['user_data']
            return self.test_response(httplib.ACCEPTED, json.dumps(node))

    def _1_0_servers_srv_xvpn7(self, method, url, body, headers):
        if method == 'DELETE':
            return self.test_response(httplib.ACCEPTED, '')

    def _1_0_server_types(self, method, url, body, headers):
        if method == 'GET':
            return self.test_response(httplib.OK, self.fixtures.load('list_server_types.json'))

    def _1_0_zones(self, method, url, body, headers):
        if method == 'GET':
            if headers['Host'] == 'api.gbt.brightbox.com':
                return self.test_response(httplib.OK, '{}')
            else:
                return self.test_response(httplib.OK, self.fixtures.load('list_zones.json'))

    def _2_0_zones(self, method, url, body, headers):
        data = '{"error_name":"unrecognised_endpoint", "errors": ["The request was for an unrecognised API endpoint"]}'
        return self.test_response(httplib.BAD_REQUEST, data)

    def _1_0_cloud_ips(self, method, url, body, headers):
        if method == 'GET':
            return self.test_response(httplib.OK, self.fixtures.load('list_cloud_ips.json'))
        elif method == 'POST':
            if body:
                body = json.loads(body)
            node = json.loads(self.fixtures.load('create_cloud_ip.json'))
            if 'reverse_dns' in body:
                node['reverse_dns'] = body['reverse_dns']
            return self.test_response(httplib.ACCEPTED, json.dumps(node))

    def _1_0_cloud_ips_cip_jsjc5(self, method, url, body, headers):
        if method == 'DELETE':
            return self.test_response(httplib.OK, '')
        elif method == 'PUT':
            body = json.loads(body)
            if body.get('reverse_dns', None) == 'fred.co.uk':
                return self.test_response(httplib.OK, '')
            else:
                return self.test_response(httplib.BAD_REQUEST, '{"error_name":"bad dns", "errors": ["Bad dns"]}')

    def _1_0_cloud_ips_cip_jsjc5_map(self, method, url, body, headers):
        if method == 'POST':
            body = json.loads(body)
            if 'destination' in body:
                return self.test_response(httplib.ACCEPTED, '')
            else:
                data = '{"error_name":"bad destination", "errors": ["Bad destination"]}'
                return self.test_response(httplib.BAD_REQUEST, data)

    def _1_0_cloud_ips_cip_jsjc5_unmap(self, method, url, body, headers):
        if method == 'POST':
            return self.test_response(httplib.ACCEPTED, '')

    def test_response(self, status, body):
        return (status, body, {'content-type': 'application/json'}, httplib.responses[status])