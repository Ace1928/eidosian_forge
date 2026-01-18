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
def test_should_retry_rate_limited_errors_until_success(self):
    count = 0

    def succeed_on_second(*_, **__) -> mock.MagicMock:
        nonlocal count
        count += 1
        if count > 1:
            successful_response = mock.MagicMock()
            successful_response.status_code = 200
            return successful_response
        else:
            raise RateLimitReachedError()
    self.driver1.connection.connection.session.send = Mock(side_effect=succeed_on_second)
    uploaded_object = self.driver1._upload_object(object_name='some name', content_type='something', request_path='/', stream=iter([]))
    self.assertEqual(True, uploaded_object['response'].success(), 'Expected to have successful response after retry')