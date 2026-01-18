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
def test_purge_by_user_type(self):
    hints = None
    users = PROVIDERS.identity_api.driver.list_users(hints)
    group_ref = {'id': uuid.uuid4().hex, 'name': uuid.uuid4().hex, 'domain_id': CONF.identity.default_domain_id}
    PROVIDERS.identity_api.driver.create_group(group_ref['id'], group_ref)
    PROVIDERS.identity_api.list_groups()
    for user in users:
        local_entity = {'domain_id': CONF.identity.default_domain_id, 'local_id': user['id'], 'entity_type': identity_mapping.EntityType.USER}
        self.assertIsNotNone(PROVIDERS.id_mapping_api.get_public_id(local_entity))
    group_entity = {'domain_id': CONF.identity.default_domain_id, 'local_id': group_ref['id'], 'entity_type': identity_mapping.EntityType.GROUP}
    self.assertIsNotNone(PROVIDERS.id_mapping_api.get_public_id(group_entity))
    provider_api.ProviderAPIs._clear_registry_instances()
    cli.MappingPurge.main()
    for user in users:
        local_entity = {'domain_id': CONF.identity.default_domain_id, 'local_id': user['id'], 'entity_type': identity_mapping.EntityType.USER}
        self.assertIsNone(PROVIDERS.id_mapping_api.get_public_id(local_entity))
    self.assertIsNotNone(PROVIDERS.id_mapping_api.get_public_id(group_entity))