import copy
import hashlib
from unittest import mock
import uuid
import fixtures
import http.client
import webtest
from keystone.auth import core as auth_core
from keystone.common import authorization
from keystone.common import context as keystone_context
from keystone.common import provider_api
from keystone.common import tokenless_auth
import keystone.conf
from keystone import exception
from keystone.federation import constants as federation_constants
from keystone.server.flask.request_processing.middleware import auth_context
from keystone.tests import unit
from keystone.tests.unit import mapping_fixtures
from keystone.tests.unit import test_backend_sql
def _assert_tokenless_auth_context(self, context, ephemeral_user=False):
    self.assertIsNotNone(context)
    self.assertEqual(self.project_id, context['project_id'])
    self.assertIn(self.role_name, context['roles'])
    if ephemeral_user:
        self.assertEqual(self.group['id'], context['group_ids'][0])
        self.assertEqual('ephemeral', context[federation_constants.PROTOCOL])
        self.assertEqual(self.idp_id, context[federation_constants.IDENTITY_PROVIDER])
    else:
        self.assertEqual(self.user['id'], context['user_id'])