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
def test_upload_object_via_stream_with_cors_headers(self):
    """
        Test we can add some ``Cross-origin resource sharing`` headers
        to the request about to be sent.
        """
    cors_headers = {'Access-Control-Allow-Origin': 'http://mozilla.com', 'Origin': 'http://storage.clouddrive.com'}
    expected_headers = {'Content-Type': 'application/octet-stream'}
    expected_headers.update(cors_headers)

    def intercept_request(request_path, method=None, data=None, headers=None, raw=True):
        self.assertDictEqual(expected_headers, headers)
        raise NotImplementedError('oops')
    self.driver.connection.request = intercept_request
    container = Container(name='CORS', extra={}, driver=self.driver)
    try:
        self.driver.upload_object_via_stream(iterator=iter(b'blob data like an image or video'), container=container, object_name='test_object', headers=cors_headers)
    except NotImplementedError:
        pass
    else:
        self.fail('Expected NotImplementedError to be thrown to verify we actually checked the expected headers')