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
class GoogleBaseConnectionTest(GoogleTestCase):
    """
    Tests for GoogleBaseConnection
    """

    def setUp(self):
        GoogleBaseAuthConnection.conn_class = GoogleAuthMockHttp
        self.mock_scopes = ['https://www.googleapis.com/auth/foo']
        kwargs = {'scopes': self.mock_scopes, 'auth_type': GoogleAuthType.IA}
        self.conn = GoogleBaseConnection(*GCE_PARAMS, **kwargs)

    def test_add_default_headers(self):
        old_headers = {}
        new_expected_headers = {'Content-Type': 'application/json', 'Host': 'www.googleapis.com'}
        new_headers = self.conn.add_default_headers(old_headers)
        self.assertEqual(new_headers, new_expected_headers)

    def test_pre_connect_hook(self):
        old_params = {}
        old_headers = {}
        auth_str = '{} {}'.format(STUB_TOKEN_FROM_FILE['token_type'], STUB_TOKEN_FROM_FILE['access_token'])
        new_expected_params = {}
        new_expected_headers = {'Authorization': auth_str}
        new_params, new_headers = self.conn.pre_connect_hook(old_params, old_headers)
        self.assertEqual(new_params, new_expected_params)
        self.assertEqual(new_headers, new_expected_headers)

    def test_encode_data(self):
        data = {'key': 'value'}
        json_data = '{"key": "value"}'
        encoded_data = self.conn.encode_data(data)
        self.assertEqual(encoded_data, json_data)

    def test_has_completed(self):
        body1 = {'endTime': '2013-06-26T10:05:07.630-07:00', 'id': '3681664092089171723', 'kind': 'compute#operation', 'status': 'DONE', 'targetId': '16211908079305042870'}
        body2 = {'endTime': '2013-06-26T10:05:07.630-07:00', 'id': '3681664092089171723', 'kind': 'compute#operation', 'status': 'RUNNING', 'targetId': '16211908079305042870'}
        response1 = MockJsonResponse(body1)
        response2 = MockJsonResponse(body2)
        self.assertTrue(self.conn.has_completed(response1))
        self.assertFalse(self.conn.has_completed(response2))

    def test_get_poll_request_kwargs(self):
        body = {'endTime': '2013-06-26T10:05:07.630-07:00', 'id': '3681664092089171723', 'kind': 'compute#operation', 'selfLink': 'https://www.googleapis.com/operations-test'}
        response = MockJsonResponse(body)
        expected_kwargs = {'action': 'https://www.googleapis.com/operations-test'}
        kwargs = self.conn.get_poll_request_kwargs(response, None, {})
        self.assertEqual(kwargs, expected_kwargs)

    def test_morph_action_hook(self):
        self.conn.request_path = '/compute/apiver/project/project-name'
        action1 = 'https://www.googleapis.com/compute/apiver/project/project-name/instances'
        action2 = '/instances'
        expected_request = '/compute/apiver/project/project-name/instances'
        request1 = self.conn.morph_action_hook(action1)
        request2 = self.conn.morph_action_hook(action2)
        self.assertEqual(request1, expected_request)
        self.assertEqual(request2, expected_request)