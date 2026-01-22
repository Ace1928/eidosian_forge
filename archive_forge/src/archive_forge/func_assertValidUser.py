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
def assertValidUser(self, entity, ref=None):
    self.assertIsNotNone(entity.get('domain_id'))
    self.assertIsNotNone(entity.get('email'))
    self.assertNotIn('password', entity)
    self.assertNotIn('projectId', entity)
    self.assertIn('password_expires_at', entity)
    if ref:
        self.assertEqual(ref['domain_id'], entity['domain_id'])
        self.assertEqual(ref['email'], entity['email'])
        if 'default_project_id' in ref:
            self.assertIsNotNone(ref['default_project_id'])
            self.assertEqual(ref['default_project_id'], entity['default_project_id'])
    return entity