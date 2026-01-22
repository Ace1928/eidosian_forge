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
class TestMappingPurge(unit.SQLDriverOverrides, unit.BaseTestCase):

    class FakeConfCommand(object):

        def __init__(self, parent):
            self.extension = False
            self.all = parent.command_all
            self.type = parent.command_type
            self.domain_name = parent.command_domain_name
            self.local_id = parent.command_local_id
            self.public_id = parent.command_public_id

    def setUp(self):
        super(TestMappingPurge, self).setUp()
        self.config_fixture = self.useFixture(oslo_config.fixture.Config(CONF))
        self.config_fixture.register_cli_opt(cli.command_opt)
        parser_test = argparse.ArgumentParser()
        subparsers = parser_test.add_subparsers()
        self.parser = cli.MappingPurge.add_argument_parser(subparsers)

    def test_mapping_purge_with_no_arguments_fails(self):
        self.command_type = None
        self.command_all = False
        self.command_domain_name = None
        self.command_local_id = None
        self.command_public_id = None
        self.useFixture(fixtures.MockPatchObject(CONF, 'command', self.FakeConfCommand(self)))
        self.assertRaises(ValueError, cli.MappingPurge.main)

    def test_mapping_purge_with_all_and_other_argument_fails(self):
        self.command_type = 'user'
        self.command_all = True
        self.command_domain_name = None
        self.command_local_id = None
        self.command_public_id = None
        self.useFixture(fixtures.MockPatchObject(CONF, 'command', self.FakeConfCommand(self)))
        self.assertRaises(ValueError, cli.MappingPurge.main)

    def test_mapping_purge_with_only_all_passes(self):
        args = ['--all']
        res = self.parser.parse_args(args)
        self.assertTrue(vars(res)['all'])

    def test_mapping_purge_with_domain_name_argument_succeeds(self):
        args = ['--domain-name', uuid.uuid4().hex]
        self.parser.parse_args(args)

    def test_mapping_purge_with_public_id_argument_succeeds(self):
        args = ['--public-id', uuid.uuid4().hex]
        self.parser.parse_args(args)

    def test_mapping_purge_with_local_id_argument_succeeds(self):
        args = ['--local-id', uuid.uuid4().hex]
        self.parser.parse_args(args)

    def test_mapping_purge_with_type_argument_succeeds(self):
        args = ['--type', 'user']
        self.parser.parse_args(args)
        args = ['--type', 'group']
        self.parser.parse_args(args)

    def test_mapping_purge_with_invalid_argument_fails(self):
        args = ['--invalid-option', 'some value']
        self.assertRaises(unit.UnexpectedExit, self.parser.parse_args, args)

    def test_mapping_purge_with_all_other_combinations_passes(self):
        args = ['--type', 'user', '--local-id', uuid.uuid4().hex]
        self.parser.parse_args(args)
        args.append('--domain-name')
        args.append('test')
        self.parser.parse_args(args)
        args.append('--public-id')
        args.append(uuid.uuid4().hex)
        self.parser.parse_args(args)

    @mock.patch.object(keystone.identity.MappingManager, 'purge_mappings')
    def test_mapping_purge_type_user(self, purge_mock):
        self.command_type = 'user'
        self.command_all = False
        self.command_domain_name = None
        self.command_local_id = uuid.uuid4().hex
        self.command_public_id = uuid.uuid4().hex
        self.useFixture(fixtures.MockPatchObject(CONF, 'command', self.FakeConfCommand(self)))

        def fake_load_backends():
            return dict(id_mapping_api=keystone.identity.core.MappingManager, resource_api=None)
        self.useFixture(fixtures.MockPatch('keystone.server.backends.load_backends', side_effect=fake_load_backends))
        cli.MappingPurge.main()
        purge_mock.assert_called_with({'entity_type': 'user', 'local_id': self.command_local_id, 'public_id': self.command_public_id})