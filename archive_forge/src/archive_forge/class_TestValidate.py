import base64
import datetime
import hashlib
import os
from unittest import mock
import uuid
from oslo_utils import timeutils
from keystone.common import fernet_utils
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.identity.backends import resource_options as ro
from keystone.receipt.providers import fernet
from keystone.receipt import receipt_formatters
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
from keystone.token import provider as token_provider
class TestValidate(unit.TestCase):

    def setUp(self):
        super(TestValidate, self).setUp()
        self.useFixture(database.Database())
        self.useFixture(ksfixtures.ConfigAuthPlugins(self.config_fixture, ['totp', 'token', 'password']))
        self.load_backends()
        PROVIDERS.resource_api.create_domain(default_fixtures.ROOT_DOMAIN['id'], default_fixtures.ROOT_DOMAIN)

    def config_overrides(self):
        super(TestValidate, self).config_overrides()
        self.config_fixture.config(group='receipt', provider='fernet')

    def test_validate_v3_receipt_simple(self):
        domain_ref = unit.new_domain_ref()
        domain_ref = PROVIDERS.resource_api.create_domain(domain_ref['id'], domain_ref)
        rule_list = [['password', 'totp'], ['password', 'totp', 'token']]
        user_ref = unit.new_user_ref(domain_ref['id'])
        user_ref = PROVIDERS.identity_api.create_user(user_ref)
        user_ref['options'][ro.MFA_RULES_OPT.option_name] = rule_list
        user_ref['options'][ro.MFA_ENABLED_OPT.option_name] = True
        PROVIDERS.identity_api.update_user(user_ref['id'], user_ref)
        method_names = ['password']
        receipt = PROVIDERS.receipt_provider_api.issue_receipt(user_ref['id'], method_names)
        receipt = PROVIDERS.receipt_provider_api.validate_receipt(receipt.id)
        self.assertIsInstance(receipt.expires_at, str)
        self.assertIsInstance(receipt.issued_at, str)
        self.assertEqual(set(method_names), set(receipt.methods))
        self.assertEqual(set((frozenset(r) for r in rule_list)), set((frozenset(r) for r in receipt.required_methods)))
        self.assertEqual(user_ref['id'], receipt.user_id)

    def test_validate_v3_receipt_validation_error_exc(self):
        receipt_id = uuid.uuid4().hex
        self.assertRaises(exception.ReceiptNotFound, PROVIDERS.receipt_provider_api.validate_receipt, receipt_id)