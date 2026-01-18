import datetime
import os
import time
from unittest import mock
import uuid
import fixtures
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import fixture
from keystoneauth1 import loading
from keystoneauth1 import session
import oslo_cache
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import timeutils
import pbr.version
import testresources
from testtools import matchers
import webob
import webob.dec
from keystonemiddleware import auth_token
from keystonemiddleware.auth_token import _base
from keystonemiddleware.auth_token import _cache
from keystonemiddleware.auth_token import _exceptions as ksm_exceptions
from keystonemiddleware.tests.unit.auth_token import base
from keystonemiddleware.tests.unit import client_fixtures
def test_unknown_server_versions(self):
    versions = fixture.DiscoveryList(v2=False, v3_id='v4', href=BASE_URI)
    self.set_middleware()
    self.requests_mock.get(BASE_URI, json=versions, status_code=300)
    self.call_middleware(headers={'X-Auth-Token': uuid.uuid4().hex}, expected_status=503)
    self.assertIn('versions [v3.0]', self.logger.output)