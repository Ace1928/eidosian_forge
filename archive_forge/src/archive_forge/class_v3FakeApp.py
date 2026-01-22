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
class v3FakeApp(FakeApp):
    """This represents a v3 WSGI app protected by the auth_token middleware."""

    def __init__(self, expected_env=None, need_service_token=False):
        v3_default_env_additions = dict(EXPECTED_V3_DEFAULT_ENV_ADDITIONS)
        if expected_env:
            v3_default_env_additions.update(expected_env)
        super(v3FakeApp, self).__init__(expected_env=v3_default_env_additions, need_service_token=need_service_token)