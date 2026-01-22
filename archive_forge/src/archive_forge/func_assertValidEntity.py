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
def assertValidEntity(self, entity, ref=None, keys_to_check=None):
    """Make assertions common to all API entities.

        If a reference is provided, the entity will also be compared against
        the reference.
        """
    if keys_to_check is not None:
        keys = keys_to_check
    else:
        keys = ['name', 'description', 'enabled']
    for k in ['id'] + keys:
        msg = '%s unexpectedly None in %s' % (k, entity)
        self.assertIsNotNone(entity.get(k), msg)
    self.assertIsNotNone(entity.get('links'))
    self.assertIsNotNone(entity['links'].get('self'))
    self.assertThat(entity['links']['self'], matchers.StartsWith('http://localhost'))
    self.assertIn(entity['id'], entity['links']['self'])
    if ref:
        for k in keys:
            msg = '%s not equal: %s != %s' % (k, ref[k], entity[k])
            self.assertEqual(ref[k], entity[k])
    return entity