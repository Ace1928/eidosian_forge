import os
import sys
import unittest
from unittest import mock
from libcloud.test import MockHttp  # pylint: disable-msg=E0611
from libcloud.test import make_response, generate_random_data
from libcloud.utils.py3 import b, httplib, parse_qs, urlparse
from libcloud.common.types import InvalidCredsError
from libcloud.storage.base import Object, Container
from libcloud.test.secrets import STORAGE_OSS_PARAMS
from libcloud.storage.types import (
from libcloud.test.file_fixtures import StorageFileFixtures  # pylint: disable-msg=E0611
from libcloud.storage.drivers.oss import CHUNK_SIZE, OSSConnection, OSSStorageDriver
from libcloud.storage.drivers.dummy import DummyIterator
class OSSConnectionTestCase(unittest.TestCase):

    def setUp(self):
        self.conn = OSSConnection('44CF9590006BF252F707', 'OtxrzxIsfpFjA7SwPzILwy8Bw21TLhquhboDYROV')

    def test_signature(self):
        expected = b('26NBxoKdsyly4EDv6inkoDft/yA=')
        headers = {'Content-MD5': 'ODBGOERFMDMzQTczRUY3NUE3NzA5QzdFNUYzMDQxNEM=', 'Content-Type': 'text/html', 'Expires': 'Thu, 17 Nov 2005 18:49:58 GMT', 'X-OSS-Meta-Author': 'foo@bar.com', 'X-OSS-Magic': 'abracadabra', 'Host': 'oss-example.oss-cn-hangzhou.aliyuncs.com'}
        action = '/oss-example/nelson'
        actual = OSSConnection._get_auth_signature('PUT', headers, {}, headers['Expires'], self.conn.key, action, 'x-oss-')
        self.assertEqual(expected, actual)