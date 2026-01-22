import itertools
import os
from unittest import mock
import uuid
from oslo_config import fixture as config_fixture
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import identity
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
class TestDatabaseDomainConfigs(unit.TestCase):

    def setUp(self):
        super(TestDatabaseDomainConfigs, self).setUp()
        self.useFixture(database.Database())
        self.load_backends()
        PROVIDERS.resource_api.create_domain(default_fixtures.ROOT_DOMAIN['id'], default_fixtures.ROOT_DOMAIN)

    def test_domain_config_in_database_disabled_by_default(self):
        self.assertFalse(CONF.identity.domain_configurations_from_database)

    def test_loading_config_from_database(self):
        self.config_fixture.config(domain_configurations_from_database=True, group='identity')
        domain = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain['id'], domain)
        conf = {'ldap': {'url': uuid.uuid4().hex, 'suffix': uuid.uuid4().hex, 'use_tls': True}, 'identity': {'driver': 'ldap'}}
        PROVIDERS.domain_config_api.create_config(domain['id'], conf)
        fake_standard_driver = None
        domain_config = identity.DomainConfigs()
        domain_config.setup_domain_drivers(fake_standard_driver, PROVIDERS.resource_api)
        res = domain_config.get_domain_conf(domain['id'])
        self.assertEqual(conf['ldap']['url'], res.ldap.url)
        self.assertEqual(conf['ldap']['suffix'], res.ldap.suffix)
        self.assertEqual(CONF.ldap.query_scope, res.ldap.query_scope)
        use_tls_type = type(CONF.ldap.use_tls)
        self.assertEqual(use_tls_type(conf['ldap']['use_tls']), res.ldap.use_tls)
        self.config_fixture.config(group='identity', domain_configurations_from_database=False)
        domain_config = identity.DomainConfigs()
        domain_config.setup_domain_drivers(fake_standard_driver, PROVIDERS.resource_api)
        res = domain_config.get_domain_conf(domain['id'])
        self.assertEqual(CONF.ldap.url, res.ldap.url)
        self.assertEqual(CONF.ldap.suffix, res.ldap.suffix)
        self.assertEqual(CONF.ldap.use_tls, res.ldap.use_tls)
        self.assertEqual(CONF.ldap.query_scope, res.ldap.query_scope)