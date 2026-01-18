import datetime
import io
import itertools
import json
import logging
import sys
from unittest import mock
import uuid
from oslo_utils import encodeutils
import requests
import requests.auth
from testtools import matchers
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import plugin
from keystoneauth1 import session as client_session
from keystoneauth1.tests.unit import utils
from keystoneauth1 import token_endpoint
def test_setting_variables_on_get_endpoint(self):
    adpt = self._create_loaded_adapter()
    url = adpt.get_endpoint()
    self.assertEqual(self.TEST_URL, url)
    self._verify_endpoint_called(adpt)