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
def _test_validate_v3_token_federted_info(self, group_ids):
    domain_ref = unit.new_domain_ref()
    domain_ref = PROVIDERS.resource_api.create_domain(domain_ref['id'], domain_ref)
    user_ref = unit.new_user_ref(domain_ref['id'])
    user_ref = PROVIDERS.identity_api.create_user(user_ref)
    method_names = ['mapped']
    idp_id = uuid.uuid4().hex
    idp_ref = {'id': idp_id, 'description': uuid.uuid4().hex, 'enabled': True}
    self.federation_api.create_idp(idp_id, idp_ref)
    protocol = uuid.uuid4().hex
    auth_context_params = {'user_id': user_ref['id'], 'user_name': user_ref['name'], 'group_ids': group_ids, federation_constants.IDENTITY_PROVIDER: idp_id, federation_constants.PROTOCOL: protocol}
    auth_context = auth.core.AuthContext(**auth_context_params)
    token = PROVIDERS.token_provider_api.issue_token(user_ref['id'], method_names, auth_context=auth_context)
    token = PROVIDERS.token_provider_api.validate_token(token.id)
    self.assertEqual(user_ref['id'], token.user_id)
    self.assertEqual(user_ref['name'], token.user['name'])
    self.assertDictEqual(domain_ref, token.user_domain)
    exp_group_ids = [{'id': group_id} for group_id in group_ids]
    self.assertEqual(exp_group_ids, token.federated_groups)
    self.assertEqual(idp_id, token.identity_provider_id)
    self.assertEqual(protocol, token.protocol_id)