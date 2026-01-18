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
def test_retry_url_uses_proxy(self):
    self.set_normal_response('normal response')
    self.set_no_proxy_allowed_response('no proxy response')
    response = retry_url('http://10.10.10.10/foo', num_retries=1)
    self.assertEqual(response, 'no proxy response')