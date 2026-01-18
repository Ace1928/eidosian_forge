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
def token_response(self, request, context):
    auth_id = request.headers.get('X-Auth-Token')
    token_id = request.headers.get('X-Subject-Token')
    self.assertEqual(auth_id, FAKE_ADMIN_TOKEN_ID)
    status = 200
    response = ''
    if token_id == ERROR_TOKEN:
        msg = 'Network connection refused.'
        raise ksa_exceptions.ConnectFailure(msg)
    elif token_id == TIMEOUT_TOKEN:
        request_timeout_response(request, context)
    try:
        response = self.examples.JSON_TOKEN_RESPONSES[token_id]
    except KeyError:
        status = 404
    context.status_code = status
    return response