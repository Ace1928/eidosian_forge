import copy
import datetime
import logging
import os
from unittest import mock
import uuid
import argparse
import configparser
import fixtures
import freezegun
import http.client
import oslo_config.fixture
from oslo_log import log
from oslo_serialization import jsonutils
from oslo_upgradecheck import upgradecheck
from testtools import matchers
from keystone.cmd import cli
from keystone.cmd.doctor import caching
from keystone.cmd.doctor import credential
from keystone.cmd.doctor import database as doc_database
from keystone.cmd.doctor import debug
from keystone.cmd.doctor import federation
from keystone.cmd.doctor import ldap
from keystone.cmd.doctor import security_compliance
from keystone.cmd.doctor import tokens
from keystone.cmd.doctor import tokens_fernet
from keystone.cmd import status
from keystone.common import provider_api
from keystone.common.sql import upgrades
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.identity.mapping_backends import mapping as identity_mapping
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit.ksfixtures import ldapdb
from keystone.tests.unit.ksfixtures import policy
from keystone.tests.unit.ksfixtures import temporaryfile
from keystone.tests.unit import mapping_fixtures
class CliStatusTestCase(unit.SQLDriverOverrides, unit.TestCase):

    def setUp(self):
        self.useFixture(database.Database())
        super(CliStatusTestCase, self).setUp()
        self.load_backends()
        self.policy_file = self.useFixture(temporaryfile.SecureTempFile())
        self.policy_file_name = self.policy_file.file_name
        self.useFixture(policy.Policy(self.config_fixture, policy_file=self.policy_file_name))
        self.checks = status.Checks()

    def test_check_safe_trust_policies(self):
        with open(self.policy_file_name, 'w') as f:
            overridden_policies = {'identity:list_trusts': '', 'identity:delete_trust': '', 'identity:get_trust': '', 'identity:list_roles_for_trust': '', 'identity:get_role_for_trust': ''}
            f.write(jsonutils.dumps(overridden_policies))
        result = self.checks.check_trust_policies_are_not_empty()
        self.assertEqual(upgradecheck.Code.FAILURE, result.code)
        with open(self.policy_file_name, 'w') as f:
            overridden_policies = {'identity:list_trusts': 'rule:admin_required', 'identity:delete_trust': 'rule:admin_required', 'identity:get_trust': 'rule:admin_required', 'identity:list_roles_for_trust': 'rule:admin_required', 'identity:get_role_for_trust': 'rule:admin_required'}
            f.write(jsonutils.dumps(overridden_policies))
        result = self.checks.check_trust_policies_are_not_empty()
        self.assertEqual(upgradecheck.Code.SUCCESS, result.code)
        with open(self.policy_file_name, 'w') as f:
            overridden_policies = {}
            f.write(jsonutils.dumps(overridden_policies))
        result = self.checks.check_trust_policies_are_not_empty()
        self.assertEqual(upgradecheck.Code.SUCCESS, result.code)

    def test_check_immutable_roles(self):
        role_ref = unit.new_role_ref(name='admin')
        PROVIDERS.role_api.create_role(role_ref['id'], role_ref)
        result = self.checks.check_default_roles_are_immutable()
        self.assertEqual(upgradecheck.Code.FAILURE, result.code)
        role_ref['options'] = {'immutable': True}
        PROVIDERS.role_api.update_role(role_ref['id'], role_ref)
        result = self.checks.check_default_roles_are_immutable()
        self.assertEqual(upgradecheck.Code.SUCCESS, result.code)
        PROVIDERS.resource_api.create_domain(default_fixtures.ROOT_DOMAIN['id'], default_fixtures.ROOT_DOMAIN)
        domain_ref = unit.new_domain_ref()
        domain = PROVIDERS.resource_api.create_domain(domain_ref['id'], domain_ref)
        role_ref = unit.new_role_ref(name='admin', domain_id=domain['id'])
        PROVIDERS.role_api.create_role(role_ref['id'], role_ref)
        result = self.checks.check_default_roles_are_immutable()
        self.assertEqual(upgradecheck.Code.SUCCESS, result.code)