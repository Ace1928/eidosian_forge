import json
import unittest
from six.moves import http_client
from six.moves import urllib
import oauth2client
from oauth2client import client
from oauth2client import transport
from oauth2client.contrib import gce
class TestComputeEngine(unittest.TestCase):

    def test_application_default(self):
        default_creds = client.GoogleCredentials.get_application_default()
        self.assertIsInstance(default_creds, gce.AppAssertionCredentials)

    def test_token_info(self):
        credentials = gce.AppAssertionCredentials([])
        http = transport.get_http_object()
        self.assertIsNone(credentials.access_token)
        credentials.refresh(http)
        self.assertIsNotNone(credentials.access_token)
        query_params = {'access_token': credentials.access_token}
        token_uri = oauth2client.GOOGLE_TOKEN_INFO_URI + '?' + urllib.parse.urlencode(query_params)
        response, content = transport.request(http, token_uri)
        self.assertEqual(response.status, http_client.OK)
        content = content.decode('utf-8')
        payload = json.loads(content)
        self.assertEqual(payload['access_type'], 'offline')
        self.assertLessEqual(int(payload['expires_in']), 3600)