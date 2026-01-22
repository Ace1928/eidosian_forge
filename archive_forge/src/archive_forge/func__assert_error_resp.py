from base64 import b64encode
from cryptography.hazmat.primitives.serialization import Encoding
import fixtures
import http
from http import client
from oslo_log import log
from oslo_serialization import jsonutils
from unittest import mock
from urllib import parse
from keystone.api.os_oauth2 import AccessTokenResource
from keystone.common import provider_api
from keystone.common import utils
from keystone import conf
from keystone import exception
from keystone.federation.utils import RuleProcessor
from keystone.tests import unit
from keystone.tests.unit import test_v3
from keystone.token.provider import Manager
def _assert_error_resp(self, error_resp, error_msg, error_description):
    resp_keys = ('error', 'error_description')
    for key in resp_keys:
        self.assertIsNotNone(error_resp.get(key, None))
    self.assertEqual(error_msg, error_resp.get('error'))
    self.assertEqual(error_description, error_resp.get('error_description'))