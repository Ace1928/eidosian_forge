import os
import sys
import time
import random
import urllib
import datetime
import unittest
import threading
from unittest import mock
import requests
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib
from libcloud.common.google import (
class GoogleBaseAuthConnectionTest(GoogleTestCase):
    """
    Tests for GoogleBaseAuthConnection
    """

    def setUp(self):
        GoogleBaseAuthConnection.conn_class = GoogleAuthMockHttp
        self.mock_scopes = ['foo', 'bar']
        kwargs = {'scopes': self.mock_scopes}
        self.conn = GoogleInstalledAppAuthConnection(*GCE_PARAMS, **kwargs)

    def test_scopes(self):
        self.assertEqual(self.conn.scopes, 'foo bar')

    def test_add_default_headers(self):
        old_headers = {}
        expected_headers = {'Content-Type': 'application/x-www-form-urlencoded', 'Host': 'accounts.google.com'}
        new_headers = self.conn.add_default_headers(old_headers)
        self.assertEqual(new_headers, expected_headers)

    def test_token_request(self):
        request_body = {'code': 'asdf', 'client_id': self.conn.user_id, 'client_secret': self.conn.key, 'redirect_uri': self.conn.redirect_uri, 'grant_type': 'authorization_code'}
        new_token = self.conn._token_request(request_body)
        self.assertEqual(new_token['access_token'], STUB_IA_TOKEN['access_token'])
        exp = STUB_UTCNOW + datetime.timedelta(seconds=STUB_IA_TOKEN['expires_in'])
        self.assertEqual(new_token['expire_time'], _utc_timestamp(exp))