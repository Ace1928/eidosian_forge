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
class CachingDoctorTests(unit.TestCase):

    def test_symptom_caching_disabled(self):
        self.config_fixture.config(group='cache', enabled=False)
        self.assertTrue(caching.symptom_caching_disabled())
        self.config_fixture.config(group='cache', enabled=True)
        self.assertFalse(caching.symptom_caching_disabled())

    def test_caching_symptom_caching_enabled_without_a_backend(self):
        self.config_fixture.config(group='cache', enabled=True)
        self.config_fixture.config(group='cache', backend='dogpile.cache.null')
        self.assertTrue(caching.symptom_caching_enabled_without_a_backend())
        self.config_fixture.config(group='cache', enabled=False)
        self.config_fixture.config(group='cache', backend='dogpile.cache.null')
        self.assertFalse(caching.symptom_caching_enabled_without_a_backend())
        self.config_fixture.config(group='cache', enabled=False)
        self.config_fixture.config(group='cache', backend='dogpile.cache.memory')
        self.assertFalse(caching.symptom_caching_enabled_without_a_backend())
        self.config_fixture.config(group='cache', enabled=True)
        self.config_fixture.config(group='cache', backend='dogpile.cache.memory')
        self.assertFalse(caching.symptom_caching_enabled_without_a_backend())

    @mock.patch('keystone.cmd.doctor.caching.cache.CACHE_REGION')
    def test_symptom_connection_to_memcached(self, cache_mock):
        self.config_fixture.config(group='cache', enabled=True)
        self.config_fixture.config(group='cache', memcache_servers=['alpha.com:11211', 'beta.com:11211'])
        self.config_fixture.config(group='cache', backend='dogpile.cache.memcached')
        cache_mock.actual_backend.client.get_stats.return_value = [('alpha.com', {}), ('beta.com', {})]
        self.assertFalse(caching.symptom_connection_to_memcached())
        cache_mock.actual_backend.client.get_stats.return_value = []
        self.assertTrue(caching.symptom_connection_to_memcached())
        cache_mock.actual_backend.client.get_stats.return_value = [('alpha.com', {})]
        self.assertTrue(caching.symptom_connection_to_memcached())
        self.config_fixture.config(group='cache', memcache_servers=['alpha.com:11211', 'beta.com:11211'])
        self.config_fixture.config(group='cache', backend='oslo_cache.memcache_pool')
        cache_mock.actual_backend.client.get_stats.return_value = [('alpha.com', {}), ('beta.com', {})]
        self.assertFalse(caching.symptom_connection_to_memcached())
        cache_mock.actual_backend.client.get_stats.return_value = []
        self.assertTrue(caching.symptom_connection_to_memcached())
        cache_mock.actual_backend.client.get_stats.return_value = [('alpha.com', {})]
        self.assertTrue(caching.symptom_connection_to_memcached())