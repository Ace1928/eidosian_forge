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
def test_upload_small_object_with_glacier_ir(self):
    if self.driver.supports_s3_multipart_upload:
        self.mock_response_klass.type = 'MULTIPART'
    else:
        self.mock_response_klass.type = None
    container = Container(name='foo_bar_container', extra={}, driver=self.driver)
    object_name = 'foo_test_stream_data'
    storage_class = 'glacier_ir'
    iterator = BytesIO(b('234'))
    extra = {'content_type': 'text/plain'}
    obj = self.driver.upload_object_via_stream(container=container, object_name=object_name, iterator=iterator, extra=extra, ex_storage_class=storage_class)
    self.assertEqual(obj.name, object_name)
    self.assertEqual(obj.size, 3)