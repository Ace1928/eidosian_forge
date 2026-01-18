from tests.compat import mock, unittest
import datetime
import hashlib
import hmac
import locale
import time
import boto.utils
from boto.utils import Password
from boto.utils import pythonize_name
from boto.utils import _build_instance_metadata_url
from boto.utils import get_instance_userdata
from boto.utils import retry_url
from boto.utils import LazyLoadMetadata
from boto.compat import json, _thread
def test_retry_url_using_bytes_and_string_response(self):
    test_value = 'normal response'
    fake_response = mock.Mock()
    fake_response.read.return_value = test_value
    self.opener.return_value.open.return_value = fake_response
    response = retry_url('http://10.10.10.10/foo', num_retries=1)
    self.assertEqual(response, test_value)
    fake_response.read.return_value = test_value.encode('utf-8')
    self.opener.return_value.open.return_value = fake_response
    response = retry_url('http://10.10.10.10/foo', num_retries=1)
    self.assertEqual(response, test_value)