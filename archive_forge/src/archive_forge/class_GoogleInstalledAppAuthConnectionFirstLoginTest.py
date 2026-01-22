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
class GoogleInstalledAppAuthConnectionFirstLoginTest(LibcloudTestCase):

    def setUp(self):
        GoogleInstalledAppAuthConnection.conn_class = GoogleAuthMockHttp
        self.mock_scopes = ['https://www.googleapis.com/auth/foo']
        kwargs = {'scopes': self.mock_scopes}
        self.conn = GoogleInstalledAppAuthConnection(*GCE_PARAMS, **kwargs)
        self.conn.redirect_uri_port = random.randint(5000, 20000)

    def test_it_receives_the_code_that_google_sends_via_local_loopback(self):
        expected_code = '1234ABC'
        received_code = self._do_first_sign_in(expected_code=expected_code, state=self.conn._state)
        self.assertEqual(received_code, expected_code)

    def test_it_aborts_if_state_is_suspicious(self):
        received_code = self._do_first_sign_in(expected_code='1234ABC', state=self.conn._state + 'very suspicious')
        self.assertEqual(received_code, None)

    def _do_first_sign_in(self, expected_code, state):
        """
        :param expected_code: The code that the fake Google sign-in local GET request will have in its query.
        :type expected_code: `str`
        :param state: The state that the fake Google sign-in local GET request will have in its query.
        :type state: `str`
        :return: The code that was extracted through local loopback.
        :rtype: `Optional[str]`
        """
        received_code = None

        def _get_code():
            nonlocal received_code
            received_code = self.conn.get_code()

        def _send_code():
            target_url = self.conn._redirect_uri_with_port
            params = {'state': state, 'code': expected_code}
            params = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
            requests.get(url=target_url, params=params)
        fake_sign_in_thread = threading.Thread(target=_get_code)
        fake_google_response = threading.Thread(target=_send_code)
        fake_sign_in_thread.start()
        time.sleep(0.2)
        fake_google_response.start()
        fake_google_response.join()
        fake_sign_in_thread.join()
        return received_code