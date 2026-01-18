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
def test_valid_system_scoped_token_request(self):
    delta_expected_env = {'HTTP_OPENSTACK_SYSTEM_SCOPE': 'all', 'HTTP_X_PROJECT_ID': None, 'HTTP_X_PROJECT_NAME': None, 'HTTP_X_PROJECT_DOMAIN_ID': None, 'HTTP_X_PROJECT_DOMAIN_NAME': None, 'HTTP_X_TENANT_ID': None, 'HTTP_X_TENANT_NAME': None, 'HTTP_X_TENANT': None}
    self.set_middleware(expected_env=delta_expected_env)
    self.assert_valid_request_200(self.examples.v3_SYSTEM_SCOPED_TOKEN)
    self.assertLastPath('/v3/auth/tokens')