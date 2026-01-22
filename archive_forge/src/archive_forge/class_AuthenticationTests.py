import json
from unittest import mock
import requests
from cinderclient import exceptions
from cinderclient.tests.unit import utils
from cinderclient.v3 import client
class AuthenticationTests(utils.TestCase):

    def test_authenticate_success(self):
        cs = client.Client('username', 'password', 'project_id', 'auth_url')
        management_url = 'https://localhost/v2.1/443470'
        auth_response = utils.TestResponse({'status_code': 204, 'headers': {'x-server-management-url': management_url, 'x-auth-token': '1b751d74-de0c-46ae-84f0-915744b582d1'}})
        mock_request = mock.Mock(return_value=auth_response)

        @mock.patch.object(requests, 'request', mock_request)
        def test_auth_call():
            cs.client.authenticate()
            headers = {'Accept': 'application/json', 'X-Auth-User': 'username', 'X-Auth-Key': 'password', 'X-Auth-Project-Id': 'project_id', 'User-Agent': cs.client.USER_AGENT}
            mock_request.assert_called_with('GET', cs.client.auth_url, headers=headers, **self.TEST_REQUEST_BASE)
            self.assertEqual(auth_response.headers['x-server-management-url'], cs.client.management_url)
            self.assertEqual(auth_response.headers['x-auth-token'], cs.client.auth_token)
        test_auth_call()

    def test_authenticate_failure(self):
        cs = client.Client('username', 'password', 'project_id', 'auth_url')
        auth_response = utils.TestResponse({'status_code': 401})
        mock_request = mock.Mock(return_value=auth_response)

        @mock.patch.object(requests, 'request', mock_request)
        def test_auth_call():
            self.assertRaises(exceptions.Unauthorized, cs.client.authenticate)
        test_auth_call()

    def test_auth_automatic(self):
        cs = client.Client('username', 'password', 'project_id', 'auth_url')
        http_client = cs.client
        http_client.management_url = ''
        mock_request = mock.Mock(return_value=(None, None))

        @mock.patch.object(http_client, 'request', mock_request)
        @mock.patch.object(http_client, 'authenticate')
        def test_auth_call(m):
            http_client.get('/')
            self.assertTrue(m.called)
            self.assertTrue(mock_request.called)
        test_auth_call()

    def test_auth_manual(self):
        cs = client.Client('username', 'password', 'project_id', 'auth_url')

        @mock.patch.object(cs.client, 'authenticate')
        def test_auth_call(m):
            cs.authenticate()
            self.assertTrue(m.called)
        test_auth_call()