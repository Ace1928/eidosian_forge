import base64
import copy
import hashlib
import jwt.utils
import logging
import ssl
from testtools import matchers
import time
from unittest import mock
import uuid
import webob.dec
import fixtures
from oslo_config import cfg
import six
from six.moves import http_client
import testresources
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import session
from keystonemiddleware.auth_token import _cache
from keystonemiddleware import external_oauth2_token
from keystonemiddleware.tests.unit.auth_token import base
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit import client_fixtures
from keystonemiddleware.tests.unit import utils
def _introspect_response(self, request, context, auth_method=None, introspect_client_id=None, introspect_client_secret=None, access_token=None, active=True, exp_time=None, cert_thumb=None, metadata=None, status_code=200, system_scope=False):
    if auth_method == 'tls_client_auth':
        body = 'client_id=%s&token=%s&token_type_hint=access_token' % (introspect_client_id, access_token)
        self.assertEqual(request.text, body)
    elif auth_method == 'client_secret_post':
        body = 'client_id=%s&client_secret=%s&token=%s&token_type_hint=access_token' % (introspect_client_id, introspect_client_secret, access_token)
        self.assertEqual(request.text, body)
    elif auth_method == 'client_secret_basic':
        body = 'token=%s&token_type_hint=access_token' % access_token
        self.assertEqual(request.text, body)
        auth_basic = request._request.headers.get('Authorization')
        self.assertIsNotNone(auth_basic)
        auth = 'Basic ' + base64.standard_b64encode(('%s:%s' % (introspect_client_id, introspect_client_secret)).encode('ascii')).decode('ascii')
        self.assertEqual(auth_basic, auth)
    elif auth_method == 'private_key_jwt':
        self.assertIn('client_id=%s' % introspect_client_id, request.text)
        self.assertIn('client_assertion_type=urn%3Aietf%3Aparams%3Aoauth%3Aclient-assertion-type%3Ajwt-bearer', request.text)
        self.assertIn('client_assertion=', request.text)
        self.assertIn('token=%s' % access_token, request.text)
        self.assertIn('token_type_hint=access_token', request.text)
    elif auth_method == 'client_secret_jwt':
        self.assertIn('client_id=%s' % introspect_client_id, request.text)
        self.assertIn('client_assertion_type=urn%3Aietf%3Aparams%3Aoauth%3Aclient-assertion-type%3Ajwt-bearer', request.text)
        self.assertIn('client_assertion=', request.text)
        self.assertIn('token=%s' % access_token, request.text)
        self.assertIn('token_type_hint=access_token', request.text)
    resp = {'iat': 1670311634, 'jti': str(uuid.uuid4()), 'iss': str(uuid.uuid4()), 'aud': str(uuid.uuid4()), 'sub': str(uuid.uuid4()), 'typ': 'Bearer', 'azp': str(uuid.uuid4()), 'acr': '1', 'scope': 'default'}
    if system_scope:
        resp['system_scope'] = 'all'
    if exp_time is not None:
        resp['exp'] = exp_time
    else:
        resp['exp'] = time.time() + 3600
    if cert_thumb is not None:
        resp['cnf'] = {'x5t#S256': cert_thumb}
    if metadata:
        for key in metadata:
            resp[key] = metadata[key]
    if active is not None:
        resp['active'] = active
    context.status_code = status_code
    return resp