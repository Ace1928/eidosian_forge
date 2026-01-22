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
class CliBootStrapTestCase(unit.SQLDriverOverrides, unit.TestCase):

    def setUp(self):
        self.useFixture(database.Database())
        super(CliBootStrapTestCase, self).setUp()
        self.bootstrap = cli.BootStrap()

    def config_files(self):
        self.config_fixture.register_cli_opt(cli.command_opt)
        config_files = super(CliBootStrapTestCase, self).config_files()
        config_files.append(unit.dirs.tests_conf('backend_sql.conf'))
        return config_files

    def config(self, config_files):
        CONF(args=['bootstrap', '--bootstrap-password', uuid.uuid4().hex], project='keystone', default_config_files=config_files)

    def test_bootstrap(self):
        self._do_test_bootstrap(self.bootstrap)

    def _do_test_bootstrap(self, bootstrap):
        try:
            PROVIDERS.resource_api.create_domain(default_fixtures.ROOT_DOMAIN['id'], default_fixtures.ROOT_DOMAIN)
        except exception.Conflict:
            pass
        bootstrap.do_bootstrap()
        project = PROVIDERS.resource_api.get_project_by_name(bootstrap.project_name, 'default')
        user = PROVIDERS.identity_api.get_user_by_name(bootstrap.username, 'default')
        admin_role = PROVIDERS.role_api.get_role(bootstrap.role_id)
        manager_role = PROVIDERS.role_api.get_role(bootstrap.manager_role_id)
        member_role = PROVIDERS.role_api.get_role(bootstrap.member_role_id)
        reader_role = PROVIDERS.role_api.get_role(bootstrap.reader_role_id)
        service_role = PROVIDERS.role_api.get_role(bootstrap.service_role_id)
        role_list = PROVIDERS.assignment_api.get_roles_for_user_and_project(user['id'], project['id'])
        role_list_len = 5
        if bootstrap.bootstrapper.project_name:
            role_list_len = 4
        self.assertIs(role_list_len, len(role_list))
        self.assertIn(admin_role['id'], role_list)
        self.assertIn(manager_role['id'], role_list)
        self.assertIn(member_role['id'], role_list)
        self.assertIn(reader_role['id'], role_list)
        if not bootstrap.bootstrapper.project_name:
            self.assertIn(service_role['id'], role_list)
        system_roles = PROVIDERS.assignment_api.list_system_grants_for_user(user['id'])
        self.assertIs(1, len(system_roles))
        self.assertEqual(system_roles[0]['id'], admin_role['id'])
        with self.make_request():
            PROVIDERS.identity_api.authenticate(user['id'], bootstrap.password)
        if bootstrap.region_id:
            region = PROVIDERS.catalog_api.get_region(bootstrap.region_id)
            self.assertEqual(self.region_id, region['id'])
        if bootstrap.service_id:
            svc = PROVIDERS.catalog_api.get_service(bootstrap.service_id)
            self.assertEqual(self.service_name, svc['name'])
            self.assertEqual(set(['admin', 'public', 'internal']), set(bootstrap.endpoints))
            urls = {'public': self.public_url, 'internal': self.internal_url, 'admin': self.admin_url}
            for interface, url in urls.items():
                endpoint_id = bootstrap.endpoints[interface]
                endpoint = PROVIDERS.catalog_api.get_endpoint(endpoint_id)
                self.assertEqual(self.region_id, endpoint['region_id'])
                self.assertEqual(url, endpoint['url'])
                self.assertEqual(svc['id'], endpoint['service_id'])
                self.assertEqual(interface, endpoint['interface'])

    def test_bootstrap_is_idempotent_when_password_does_not_change(self):
        self._do_test_bootstrap(self.bootstrap)
        app = self.loadapp()
        v3_password_data = {'auth': {'identity': {'methods': ['password'], 'password': {'user': {'name': self.bootstrap.username, 'password': self.bootstrap.password, 'domain': {'id': CONF.identity.default_domain_id}}}}}}
        with app.test_client() as c:
            auth_response = c.post('/v3/auth/tokens', json=v3_password_data)
            token = auth_response.headers['X-Subject-Token']
        self._do_test_bootstrap(self.bootstrap)
        with app.test_client() as c:
            r = c.post('/v3/auth/tokens', json=v3_password_data)
            c.get('/v3/auth/tokens', headers={'X-Auth-Token': r.headers['X-Subject-Token'], 'X-Subject-Token': token})
        admin_role = PROVIDERS.role_api.get_role(self.bootstrap.role_id)
        reader_role = PROVIDERS.role_api.get_role(self.bootstrap.reader_role_id)
        member_role = PROVIDERS.role_api.get_role(self.bootstrap.member_role_id)
        self.assertEqual(admin_role['options'], {'immutable': True})
        self.assertEqual(member_role['options'], {'immutable': True})
        self.assertEqual(reader_role['options'], {'immutable': True})

    def test_bootstrap_is_not_idempotent_when_password_does_change(self):
        self._do_test_bootstrap(self.bootstrap)
        app = self.loadapp()
        v3_password_data = {'auth': {'identity': {'methods': ['password'], 'password': {'user': {'name': self.bootstrap.username, 'password': self.bootstrap.password, 'domain': {'id': CONF.identity.default_domain_id}}}}}}
        time = datetime.datetime.utcnow()
        with freezegun.freeze_time(time) as frozen_time:
            with app.test_client() as c:
                auth_response = c.post('/v3/auth/tokens', json=v3_password_data)
                token = auth_response.headers['X-Subject-Token']
            new_passwd = uuid.uuid4().hex
            os.environ['OS_BOOTSTRAP_PASSWORD'] = new_passwd
            self._do_test_bootstrap(self.bootstrap)
            v3_password_data['auth']['identity']['password']['user']['password'] = new_passwd
            frozen_time.tick(delta=datetime.timedelta(seconds=1))
            with app.test_client() as c:
                r = c.post('/v3/auth/tokens', json=v3_password_data)
                c.get('/v3/auth/tokens', headers={'X-Auth-Token': r.headers['X-Subject-Token'], 'X-Subject-Token': token}, expected_status_code=http.client.NOT_FOUND)

    def test_bootstrap_recovers_user(self):
        self._do_test_bootstrap(self.bootstrap)
        user_id = PROVIDERS.identity_api.get_user_by_name(self.bootstrap.username, 'default')['id']
        PROVIDERS.identity_api.update_user(user_id, {'enabled': False, 'password': uuid.uuid4().hex})
        self._do_test_bootstrap(self.bootstrap)
        with self.make_request():
            PROVIDERS.identity_api.authenticate(user_id, self.bootstrap.password)

    def test_bootstrap_with_explicit_immutable_roles(self):
        CONF(args=['bootstrap', '--bootstrap-password', uuid.uuid4().hex, '--immutable-roles'], project='keystone')
        self._do_test_bootstrap(self.bootstrap)
        admin_role = PROVIDERS.role_api.get_role(self.bootstrap.role_id)
        reader_role = PROVIDERS.role_api.get_role(self.bootstrap.reader_role_id)
        member_role = PROVIDERS.role_api.get_role(self.bootstrap.member_role_id)
        self.assertTrue(admin_role['options']['immutable'])
        self.assertTrue(member_role['options']['immutable'])
        self.assertTrue(reader_role['options']['immutable'])

    def test_bootstrap_with_default_immutable_roles(self):
        CONF(args=['bootstrap', '--bootstrap-password', uuid.uuid4().hex], project='keystone')
        self._do_test_bootstrap(self.bootstrap)
        admin_role = PROVIDERS.role_api.get_role(self.bootstrap.role_id)
        reader_role = PROVIDERS.role_api.get_role(self.bootstrap.reader_role_id)
        member_role = PROVIDERS.role_api.get_role(self.bootstrap.member_role_id)
        self.assertTrue(admin_role['options']['immutable'])
        self.assertTrue(member_role['options']['immutable'])
        self.assertTrue(reader_role['options']['immutable'])

    def test_bootstrap_with_no_immutable_roles(self):
        CONF(args=['bootstrap', '--bootstrap-password', uuid.uuid4().hex, '--no-immutable-roles'], project='keystone')
        self._do_test_bootstrap(self.bootstrap)
        admin_role = PROVIDERS.role_api.get_role(self.bootstrap.role_id)
        reader_role = PROVIDERS.role_api.get_role(self.bootstrap.reader_role_id)
        member_role = PROVIDERS.role_api.get_role(self.bootstrap.member_role_id)
        self.assertNotIn('immutable', admin_role['options'])
        self.assertNotIn('immutable', member_role['options'])
        self.assertNotIn('immutable', reader_role['options'])

    def test_bootstrap_with_ambiguous_role_names(self):
        self._do_test_bootstrap(self.bootstrap)
        domain = {'id': uuid.uuid4().hex, 'name': uuid.uuid4().hex}
        domain = PROVIDERS.resource_api.create_domain(domain['id'], domain)
        domain_roles = {}
        for name in ['admin', 'member', 'reader', 'service']:
            domain_role = {'domain_id': domain['id'], 'id': uuid.uuid4().hex, 'name': name}
            domain_roles[name] = PROVIDERS.role_api.create_role(domain_role['id'], domain_role)
            self._do_test_bootstrap(self.bootstrap)