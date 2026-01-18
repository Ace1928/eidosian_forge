import base64
import datetime
import hashlib
import os
from unittest import mock
import uuid
from oslo_utils import timeutils
from keystone.common import fernet_utils
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.identity.backends import resource_options as ro
from keystone.receipt.providers import fernet
from keystone.receipt import receipt_formatters
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
from keystone.token import provider as token_provider
def test_restore_padding(self):
    binary_to_test = [b'a', b'aa', b'aaa']
    for binary in binary_to_test:
        encoded_string = base64.urlsafe_b64encode(binary)
        encoded_string = encoded_string.decode('utf-8')
        encoded_str_without_padding = encoded_string.rstrip('=')
        self.assertFalse(encoded_str_without_padding.endswith('='))
        encoded_str_with_padding_restored = receipt_formatters.ReceiptFormatter.restore_padding(encoded_str_without_padding)
        self.assertEqual(encoded_string, encoded_str_with_padding_restored)