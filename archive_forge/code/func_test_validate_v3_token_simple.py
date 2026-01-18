import base64
import datetime
import hashlib
import os
from unittest import mock
import uuid
import fixtures
from oslo_log import log
from oslo_utils import timeutils
from keystone import auth
from keystone.common import fernet_utils
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.federation import constants as federation_constants
from keystone.models import token_model
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
from keystone.token import provider
from keystone.token.providers import fernet
from keystone.token import token_formatters
def test_validate_v3_token_simple(self):
    domain_ref = unit.new_domain_ref()
    domain_ref = PROVIDERS.resource_api.create_domain(domain_ref['id'], domain_ref)
    user_ref = unit.new_user_ref(domain_ref['id'])
    user_ref = PROVIDERS.identity_api.create_user(user_ref)
    method_names = ['password']
    token = PROVIDERS.token_provider_api.issue_token(user_ref['id'], method_names)
    token = PROVIDERS.token_provider_api.validate_token(token.id)
    self.assertIsInstance(token.audit_ids, list)
    self.assertIsInstance(token.expires_at, str)
    self.assertIsInstance(token.issued_at, str)
    self.assertEqual(method_names, token.methods)
    self.assertEqual(user_ref['id'], token.user_id)
    self.assertEqual(user_ref['name'], token.user['name'])
    self.assertDictEqual(domain_ref, token.user_domain)
    self.assertEqual(user_ref['password_expires_at'], token.user['password_expires_at'])