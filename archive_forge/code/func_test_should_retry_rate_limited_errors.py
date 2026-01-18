import sys
import errno
import hashlib
from io import BytesIO
from unittest import mock
from unittest.mock import Mock
from libcloud.test import MockHttp, BodyStream, unittest
from libcloud.utils.py3 import PY2, StringIO, b, httplib, assertRaisesRegex
from libcloud.storage.base import DEFAULT_CONTENT_TYPE, StorageDriver
from libcloud.common.exceptions import RateLimitReachedError
from libcloud.test.storage.base import BaseRangeDownloadMockHttp
@mock.patch('os.environ', {'LIBCLOUD_RETRY_FAILED_HTTP_REQUESTS': True})
def test_should_retry_rate_limited_errors(self):

    class SecondException(Exception):
        pass
    count = 0

    def raise_on_second(*_, **__):
        nonlocal count
        count += 1
        if count > 1:
            raise SecondException()
        else:
            raise RateLimitReachedError()
    send_mock = Mock()
    self.driver1.connection.connection.session.send = send_mock
    send_mock.side_effect = raise_on_second
    with self.assertRaises(SecondException):
        self.driver1._upload_object(object_name='some name', content_type='something', request_path='/', stream=iter([]))