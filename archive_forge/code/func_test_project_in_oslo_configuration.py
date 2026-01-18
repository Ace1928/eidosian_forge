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
def test_project_in_oslo_configuration(self):
    project = uuid.uuid4().hex
    project_version = uuid.uuid4().hex
    ksm_version = uuid.uuid4().hex
    conf = {'username': self.username, 'auth_url': self.auth_url}
    with mock.patch.object(self.cfg.conf, 'project', new=project, create=True):
        app = self._create_app(conf, project_version, ksm_version)
    project = '{0}/{1} '.format(project, project_version)
    self._assert_user_agent(app, project, ksm_version)