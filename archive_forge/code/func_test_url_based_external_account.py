import datetime
import json
import os
import socket
from tempfile import NamedTemporaryFile
import threading
import time
import sys
import google.auth
from google.auth import _helpers
from googleapiclient import discovery
from six.moves import BaseHTTPServer
from google.oauth2 import service_account
import pytest
from mock import patch
def test_url_based_external_account(dns_access, oidc_credentials, service_account_info):

    class TestResponseHandler(BaseHTTPServer.BaseHTTPRequestHandler):

        def do_GET(self):
            if self.headers['my-header'] != 'expected-value':
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'missing header'}).encode('utf-8'))
            elif self.path != '/token':
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'incorrect token path'}).encode('utf-8'))
            else:
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'access_token': oidc_credentials.token}).encode('utf-8'))

    class TestHTTPServer(BaseHTTPServer.HTTPServer, object):

        def __init__(self):
            self.port = self._find_open_port()
            super(TestHTTPServer, self).__init__(('', self.port), TestResponseHandler)

        @staticmethod
        def _find_open_port():
            s = socket.socket()
            s.bind(('', 0))
            return s.getsockname()[1]

        def __exit__(self, *args):
            self.shutdown()

        def __enter__(self):
            return self
    with TestHTTPServer() as server:
        threading.Thread(target=server.serve_forever).start()
        assert get_project_dns(dns_access, {'type': 'external_account', 'audience': _AUDIENCE_OIDC, 'subject_token_type': 'urn:ietf:params:oauth:token-type:jwt', 'token_url': 'https://sts.googleapis.com/v1/token', 'service_account_impersonation_url': 'https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/{}:generateAccessToken'.format(oidc_credentials.service_account_email), 'credential_source': {'url': 'http://localhost:{}/token'.format(server.port), 'headers': {'my-header': 'expected-value'}, 'format': {'type': 'json', 'subject_token_field_name': 'access_token'}}})