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
def test_s3_list_multipart_uploads(self):
    if not self.driver.supports_s3_multipart_upload:
        return
    self.mock_response_klass.type = 'LIST_MULTIPART'
    S3StorageDriver.RESPONSES_PER_REQUEST = 2
    container = Container(name='foo_bar_container', extra={}, driver=self.driver)
    for upload in self.driver.ex_iterate_multipart_uploads(container):
        self.assertNotEqual(upload.key, None)
        self.assertNotEqual(upload.id, None)
        self.assertNotEqual(upload.created_at, None)
        self.assertNotEqual(upload.owner, None)
        self.assertNotEqual(upload.initiator, None)