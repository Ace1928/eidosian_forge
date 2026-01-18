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
def test_not_of_string_type(self):
    hostnames = [None, 0, False, [], {}]
    for h in hostnames:
        result = boto.utils.host_is_ipv6(h)
        self.assertFalse(result)