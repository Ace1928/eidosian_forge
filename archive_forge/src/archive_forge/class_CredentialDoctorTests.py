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
class CredentialDoctorTests(unit.TestCase):

    def test_credential_and_fernet_key_repositories_match(self):
        directory = self.useFixture(fixtures.TempDir()).path
        self.config_fixture.config(group='credential', key_repository=directory)
        self.config_fixture.config(group='fernet_tokens', key_repository=directory)
        self.assertTrue(credential.symptom_unique_key_repositories())

    def test_credential_and_fernet_key_repositories_are_unique(self):
        self.config_fixture.config(group='credential', key_repository='/etc/keystone/cred-repo')
        self.config_fixture.config(group='fernet_tokens', key_repository='/etc/keystone/fernet-repo')
        self.assertFalse(credential.symptom_unique_key_repositories())

    @mock.patch('keystone.cmd.doctor.credential.utils')
    def test_usability_of_cred_fernet_key_repo_raised(self, mock_utils):
        self.config_fixture.config(group='credential', provider='fernet')
        mock_utils.FernetUtils().validate_key_repository.return_value = False
        self.assertTrue(credential.symptom_usability_of_credential_fernet_key_repository())

    @mock.patch('keystone.cmd.doctor.credential.utils')
    def test_usability_of_cred_fernet_key_repo_not_raised(self, mock_utils):
        self.config_fixture.config(group='credential', provider='my-driver')
        mock_utils.FernetUtils().validate_key_repository.return_value = True
        self.assertFalse(credential.symptom_usability_of_credential_fernet_key_repository())
        self.config_fixture.config(group='credential', provider='fernet')
        mock_utils.FernetUtils().validate_key_repository.return_value = True
        self.assertFalse(credential.symptom_usability_of_credential_fernet_key_repository())

    @mock.patch('keystone.cmd.doctor.credential.utils')
    def test_keys_in_credential_fernet_key_repository_raised(self, mock_utils):
        self.config_fixture.config(group='credential', provider='fernet')
        mock_utils.FernetUtils().load_keys.return_value = False
        self.assertTrue(credential.symptom_keys_in_credential_fernet_key_repository())

    @mock.patch('keystone.cmd.doctor.credential.utils')
    def test_keys_in_credential_fernet_key_repository_not_raised(self, mock_utils):
        self.config_fixture.config(group='credential', provider='my-driver')
        mock_utils.FernetUtils().load_keys.return_value = True
        self.assertFalse(credential.symptom_keys_in_credential_fernet_key_repository())
        self.config_fixture.config(group='credential', provider='fernet')
        mock_utils.FernetUtils().load_keys.return_value = True
        self.assertFalse(credential.symptom_keys_in_credential_fernet_key_repository())