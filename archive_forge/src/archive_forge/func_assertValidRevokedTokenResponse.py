import copy
import datetime
import fixtures
import itertools
import operator
import re
from unittest import mock
from urllib import parse
import uuid
from cryptography.hazmat.primitives.serialization import Encoding
import freezegun
import http.client
from oslo_serialization import jsonutils as json
from oslo_utils import fixture
from oslo_utils import timeutils
from testtools import matchers
from testtools import testcase
from keystone import auth
from keystone.auth.plugins import totp
from keystone.common import authorization
from keystone.common import provider_api
from keystone.common.rbac_enforcer import policy
from keystone.common import utils
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.identity.backends import resource_options as ro
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import test_v3
def assertValidRevokedTokenResponse(self, events_response, **kwargs):
    events = events_response['events']
    self.assertEqual(1, len(events))
    for k, v in kwargs.items():
        self.assertEqual(v, events[0].get(k))
    self.assertIsNotNone(events[0]['issued_before'])
    self.assertIsNotNone(events_response['links'])
    del events_response['events'][0]['issued_before']
    del events_response['events'][0]['revoked_at']
    del events_response['links']
    expected_response = {'events': [kwargs]}
    self.assertEqual(expected_response, events_response)