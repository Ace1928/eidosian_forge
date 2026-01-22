import sys
import unittest
from libcloud.test import LibcloudTestCase
from libcloud.common.azure import AzureConnection
class AzureConnectionTestCase(LibcloudTestCase):

    def setUp(self):
        self.conn = AzureConnection('user', 'key')

    def test_content_length_is_used_if_set(self):
        headers = {'content-length': '123'}
        method = 'PUT'
        values = self.conn._format_special_header_values(headers, method)
        self.assertEqual(values[2], '123')

    def test_content_length_is_blank_if_new_api_version(self):
        headers = {}
        method = 'PUT'
        self.conn.API_VERSION = '2018-11-09'
        values = self.conn._format_special_header_values(headers, method)
        self.assertEqual(values[2], '')

    def test_content_length_is_zero_if_write_and_old_api_version(self):
        headers = {}
        method = 'PUT'
        self.conn.API_VERSION = '2011-08-18'
        values = self.conn._format_special_header_values(headers, method)
        self.assertEqual(values[2], '0')

    def test_content_length_is_blank_if_read_and_old_api_version(self):
        headers = {}
        method = 'GET'
        self.conn.API_VERSION = '2011-08-18'
        values = self.conn._format_special_header_values(headers, method)
        self.assertEqual(values[2], '')