import datetime
import http.client
import oslo_context.context
from oslo_serialization import jsonutils
from testtools import matchers
import uuid
import webtest
from keystone.common import authorization
from keystone.common import cache
from keystone.common import provider_api
from keystone.common.validation import validators
from keystone import exception
from keystone.resource.backends import base as resource_base
from keystone.server.flask.request_processing.middleware import auth_context
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import rest
def assertValidTokenResponse(self, r, user=None, forbid_token_id=False):
    if forbid_token_id:
        self.assertNotIn('X-Subject-Token', r.headers)
    else:
        self.assertTrue(r.headers.get('X-Subject-Token'))
    token = r.result['token']
    self.assertIsNotNone(token.get('expires_at'))
    expires_at = self.assertValidISO8601ExtendedFormatDatetime(token['expires_at'])
    self.assertIsNotNone(token.get('issued_at'))
    issued_at = self.assertValidISO8601ExtendedFormatDatetime(token['issued_at'])
    self.assertLess(issued_at, expires_at)
    self.assertIn('user', token)
    self.assertIn('id', token['user'])
    self.assertIn('name', token['user'])
    self.assertIn('domain', token['user'])
    self.assertIn('id', token['user']['domain'])
    if user is not None:
        self.assertEqual(user['id'], token['user']['id'])
        self.assertEqual(user['name'], token['user']['name'])
        self.assertEqual(user['domain_id'], token['user']['domain']['id'])
    return token