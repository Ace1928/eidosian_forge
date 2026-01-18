import os
import socket
from tests.compat import mock, unittest
from httpretty import HTTPretty
from boto import UserAgent
from boto.compat import json, parse_qs
from boto.connection import AWSQueryConnection, AWSAuthConnection, HTTPRequest
from boto.exception import BotoServerError
from boto.regioninfo import RegionInfo
def test_user_agent_not_url_encoded(self):
    headers = {'Some-Header': u'should be encoded ✓', 'User-Agent': UserAgent}
    request = HTTPRequest('PUT', 'https', 'amazon.com', 443, None, None, {}, headers, 'Body')
    mock_connection = mock.Mock()

    def mock_add_auth(req, **kwargs):
        mock_connection.headers_at_auth = req.headers.copy()
    mock_connection._auth_handler.add_auth = mock_add_auth
    request.authorize(mock_connection)
    self.assertEqual(mock_connection.headers_at_auth, {'Some-Header': 'should be encoded %E2%9C%93', 'User-Agent': UserAgent})