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
def test_valid_uuid_request_with_auth_fragments(self):
    del self.conf['identity_uri']
    self.conf['auth_protocol'] = 'https'
    self.conf['auth_host'] = 'keystone.example.com'
    self.conf['auth_port'] = '1234'
    self.conf['auth_admin_prefix'] = '/testadmin'
    self.set_middleware()
    self.assert_valid_request_200(self.token_dict['uuid_token_default'])
    self.assert_valid_last_url(self.token_dict['uuid_token_default'])