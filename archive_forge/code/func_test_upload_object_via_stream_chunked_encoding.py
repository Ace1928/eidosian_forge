import os
import sys
import copy
import hmac
import math
import hashlib
import os.path  # pylint: disable-msg=W0404
from io import BytesIO
from hashlib import sha1
from unittest import mock
from unittest.mock import Mock, PropertyMock
import libcloud.utils.files
from libcloud.test import MockHttp  # pylint: disable-msg=E0611
from libcloud.test import unittest, make_response, generate_random_data
from libcloud.utils.py3 import StringIO, b, httplib, urlquote
from libcloud.utils.files import exhaust_iterator
from libcloud.common.types import MalformedResponseError
from libcloud.storage.base import CHUNK_SIZE, Object, Container
from libcloud.storage.types import (
from libcloud.test.storage.base import BaseRangeDownloadMockHttp
from libcloud.test.file_fixtures import StorageFileFixtures  # pylint: disable-msg=E0611
from libcloud.storage.drivers.cloudfiles import CloudFilesStorageDriver
@unittest.skip('Skipping as chunking is disabled in 2.0rc1')
def test_upload_object_via_stream_chunked_encoding(self):
    bytes_blob = ''.join(['\x00' for _ in range(CHUNK_SIZE + 1)])
    hex_chunk_size = ('%X' % CHUNK_SIZE).encode('utf8')
    expected = [hex_chunk_size + b'\r\n', bytes(bytes_blob[:CHUNK_SIZE].encode('utf8')), b'\r\n', b'1\r\n', bytes(bytes_blob[CHUNK_SIZE:].encode('utf8')), b'\r\n', b'0\r\n\r\n']
    logged_data = []

    class InterceptResponse(MockHttp):

        def __init__(self, connection, response=None):
            super().__init__(connection=connection, response=response)
            old_send = self.connection.connection.send

            def intercept_send(data):
                old_send(data)
                logged_data.append(data)
            self.connection.connection.send = intercept_send

        def _v1_MossoCloudFS_py3_img_or_vid2(self, method, url, body, headers):
            headers = {'etag': 'd79fb00c27b50494a463e680d459c90c'}
            headers.update(self.base_headers)
            _201 = httplib.CREATED
            return (_201, '', headers, httplib.responses[_201])
    self.driver_klass.connectionCls.rawResponseCls = InterceptResponse
    container = Container(name='py3', extra={}, driver=self.driver)
    container.upload_object_via_stream(iterator=iter(bytes_blob), object_name='img_or_vid2')
    self.assertListEqual(expected, logged_data)