from hashlib import sha1
import random
import string
import tempfile
import time
from unittest import mock
import requests_mock
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack.object_store.v1 import account
from openstack.object_store.v1 import container
from openstack.object_store.v1 import obj
from openstack.tests.unit.cloud import test_object as base_test_object
from openstack.tests.unit import test_proxy_base
class TestDownloadObject(base_test_object.BaseTestObject):

    def setUp(self):
        super(TestDownloadObject, self).setUp()
        self.the_data = b'test body'
        self.register_uris([dict(method='GET', uri=self.object_endpoint, headers={'Content-Length': str(len(self.the_data)), 'Content-Type': 'application/octet-stream', 'Accept-Ranges': 'bytes', 'Last-Modified': 'Thu, 15 Dec 2016 13:34:14 GMT', 'Etag': '"b5c454b44fbd5344793e3fb7e3850768"', 'X-Timestamp': '1481808853.65009', 'X-Trans-Id': 'tx68c2a2278f0c469bb6de1-005857ed80dfw1', 'Date': 'Mon, 19 Dec 2016 14:24:00 GMT', 'X-Static-Large-Object': 'True', 'X-Object-Meta-Mtime': '1481513709.168512'}, content=self.the_data)])

    def test_download(self):
        data = self.cloud.object_store.download_object(self.object, container=self.container)
        self.assertEqual(data, self.the_data)
        self.assert_calls()

    def test_stream(self):
        chunk_size = 2
        for index, chunk in enumerate(self.cloud.object_store.stream_object(self.object, container=self.container, chunk_size=chunk_size)):
            chunk_len = len(chunk)
            start = index * chunk_size
            end = start + chunk_len
            self.assertLessEqual(chunk_len, chunk_size)
            self.assertEqual(chunk, self.the_data[start:end])
        self.assert_calls()