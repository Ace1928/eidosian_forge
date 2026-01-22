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
def assertValidRoleAssignment(self, entity, ref=None):
    self.assertIsNotNone(entity.get('role'))
    self.assertIsNotNone(entity['role'].get('id'))
    if entity.get('user'):
        self.assertNotIn('group', entity)
        self.assertIsNotNone(entity['user'].get('id'))
    else:
        self.assertIsNotNone(entity.get('group'))
        self.assertIsNotNone(entity['group'].get('id'))
    self.assertIsNotNone(entity.get('scope'))
    if entity['scope'].get('project'):
        self.assertNotIn('domain', entity['scope'])
        self.assertIsNotNone(entity['scope']['project'].get('id'))
    elif entity['scope'].get('domain'):
        self.assertIsNotNone(entity['scope'].get('domain'))
        self.assertIsNotNone(entity['scope']['domain'].get('id'))
    else:
        self.assertIsNotNone(entity['scope'].get('system'))
        self.assertTrue(entity['scope']['system']['all'])
    self.assertIsNotNone(entity.get('links'))
    self.assertIsNotNone(entity['links'].get('assignment'))
    if ref:
        links = ref.pop('links')
        try:
            self.assertLessEqual(ref.items(), entity.items())
            self.assertIn(links['assignment'], entity['links']['assignment'])
        finally:
            if links:
                ref['links'] = links