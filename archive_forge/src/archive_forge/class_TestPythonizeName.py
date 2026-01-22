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
class TestPythonizeName(unittest.TestCase):

    def test_empty_string(self):
        self.assertEqual(pythonize_name(''), '')

    def test_all_lower_case(self):
        self.assertEqual(pythonize_name('lowercase'), 'lowercase')

    def test_all_upper_case(self):
        self.assertEqual(pythonize_name('UPPERCASE'), 'uppercase')

    def test_camel_case(self):
        self.assertEqual(pythonize_name('OriginallyCamelCased'), 'originally_camel_cased')

    def test_already_pythonized(self):
        self.assertEqual(pythonize_name('already_pythonized'), 'already_pythonized')

    def test_multiple_upper_cased_letters(self):
        self.assertEqual(pythonize_name('HTTPRequest'), 'http_request')
        self.assertEqual(pythonize_name('RequestForHTTP'), 'request_for_http')

    def test_string_with_numbers(self):
        self.assertEqual(pythonize_name('HTTPStatus200Ok'), 'http_status_200_ok')