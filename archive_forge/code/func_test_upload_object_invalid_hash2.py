import os
import sys
import hmac
import base64
import tempfile
from io import BytesIO
from hashlib import sha1
from unittest import mock
from unittest.mock import Mock, PropertyMock
import libcloud.utils.files  # NOQA: F401
from libcloud.test import MockHttp  # pylint: disable-msg=E0611  # noqa
from libcloud.test import unittest, make_response, generate_random_data
from libcloud.utils.py3 import ET, StringIO, b, httplib, parse_qs, urlparse
from libcloud.utils.files import exhaust_iterator
from libcloud.common.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.storage.base import Object, Container
from libcloud.test.secrets import STORAGE_S3_PARAMS
from libcloud.storage.types import (
from libcloud.test.storage.base import BaseRangeDownloadMockHttp
from libcloud.storage.drivers.s3 import (
from libcloud.test.file_fixtures import StorageFileFixtures  # pylint: disable-msg=E0611
def test_upload_object_invalid_hash2(self):

    def upload_file(self, object_name=None, content_type=None, request_path=None, request_method=None, headers=None, file_path=None, stream=None):
        headers = {'etag': '"hash343hhash89h932439jsaa89"'}
        return {'response': make_response(200, headers=headers), 'bytes_transferred': 1000, 'data_hash': '0cc175b9c0f1b6a831c399e269772661'}
    old_func = self.driver_type._upload_object
    self.driver_type._upload_object = upload_file
    file_path = os.path.abspath(__file__)
    container = Container(name='foo_bar_container', extra={}, driver=self.driver)
    object_name = 'foo_test_upload'
    try:
        self.driver.upload_object(file_path=file_path, container=container, object_name=object_name, verify_hash=True)
    except ObjectHashMismatchError:
        pass
    else:
        self.fail('Invalid hash was returned but an exception was not thrown')
    finally:
        self.driver_type._upload_object = old_func