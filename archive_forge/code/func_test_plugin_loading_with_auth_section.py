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
def test_plugin_loading_with_auth_section(self):
    section = 'testsection'
    username = 'testuser'
    password = 'testpass'
    loading.register_auth_conf_options(self.cfg.conf, group=section)
    opts = loading.get_auth_plugin_conf_options('password')
    self.cfg.register_opts(opts, group=section)
    loading.register_auth_conf_options(self.cfg.conf, group=_base.AUTHTOKEN_GROUP)
    self.cfg.config(auth_section=section, group=_base.AUTHTOKEN_GROUP)
    self.cfg.config(auth_type='password', auth_url=self.AUTH_URL, password=password, project_id=self.project_id, user_domain_id='userdomainid', group=section)
    conf = {'username': username}
    body = uuid.uuid4().hex
    app = self.create_simple_middleware(body=body, conf=conf)
    resp = self.good_request(app)
    self.assertEqual(body.encode(), resp.body)
    plugin = self.get_plugin(app)
    self.assertEqual(self.AUTH_URL, plugin.auth_url)
    self.assertEqual(username, plugin._username)
    self.assertEqual(password, plugin._password)
    self.assertEqual(self.project_id, plugin._project_id)