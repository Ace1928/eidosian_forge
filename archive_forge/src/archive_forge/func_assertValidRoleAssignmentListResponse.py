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
def assertValidRoleAssignmentListResponse(self, resp, expected_length=None, resource_url=None):
    entities = resp.result.get('role_assignments')
    if expected_length or expected_length == 0:
        self.assertEqual(expected_length, len(entities))
    self.assertValidListLinks(resp.result.get('links'), resource_url=resource_url)
    for entity in entities:
        self.assertIsNotNone(entity)
        self.assertValidRoleAssignment(entity)
    return entities