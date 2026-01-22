import sys
import string
import unittest
from unittest.mock import Mock, patch
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib
from libcloud.common.nfsn import NFSNConnection
class NFSNConnectionTestCase(LibcloudTestCase):

    def setUp(self):
        NFSNConnection.conn_class = NFSNMockHttp
        NFSNMockHttp.type = None
        self.driver = NFSNConnection('testid', 'testsecret')

    def test_salt_length(self):
        self.assertEqual(16, len(self.driver._salt()))

    def test_salt_is_unique(self):
        s1 = self.driver._salt()
        s2 = self.driver._salt()
        self.assertNotEqual(s1, s2)

    def test_salt_characters(self):
        """salt must be alphanumeric"""
        salt_characters = string.ascii_letters + string.digits
        for c in self.driver._salt():
            self.assertIn(c, salt_characters)

    @patch('time.time', mock_time)
    def test_timestamp(self):
        """Check that timestamp uses time.time"""
        self.assertEqual('1000000', self.driver._timestamp())

    @patch('time.time', mock_time)
    @patch('libcloud.common.nfsn.NFSNConnection._salt', mock_salt)
    def test_auth_header(self):
        """Check that X-NFSN-Authentication is set"""
        response = self.driver.request(action='/testing')
        self.assertEqual(httplib.OK, response.status)