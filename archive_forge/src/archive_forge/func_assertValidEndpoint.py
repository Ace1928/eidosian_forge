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
def assertValidEndpoint(self, entity, ref=None):
    self.assertIsNotNone(entity.get('interface'))
    self.assertIsNotNone(entity.get('service_id'))
    self.assertIsInstance(entity['enabled'], bool)
    self.assertNotIn('legacy_endpoint_id', entity)
    if ref:
        self.assertEqual(ref['interface'], entity['interface'])
        self.assertEqual(ref['service_id'], entity['service_id'])
        if ref.get('region') is not None:
            self.assertEqual(ref['region_id'], entity.get('region_id'))
    return entity