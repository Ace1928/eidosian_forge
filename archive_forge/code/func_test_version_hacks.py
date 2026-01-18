import json
import re
from unittest import mock
from testtools import matchers
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1 import http_basic
from keystoneauth1 import noauth
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
from keystoneauth1 import token_endpoint
def test_version_hacks(self):
    self.assertEqual(self.BASE_URL, self.hacks.get_discover_hack(self.IDENTITY, self.V2_URL))
    self.assertEqual(self.BASE_URL, self.hacks.get_discover_hack(self.IDENTITY, self.V2_URL + '/'))
    self.assertEqual(self.OTHER_URL, self.hacks.get_discover_hack(self.IDENTITY, self.OTHER_URL))