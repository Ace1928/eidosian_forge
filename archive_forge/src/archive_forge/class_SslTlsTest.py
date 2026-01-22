import os
import tempfile
from unittest import mock
import uuid
import fixtures
import ldap.dn
from oslo_config import fixture as config_fixture
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception as ks_exception
from keystone.identity.backends.ldap import common as common_ldap
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import fakeldap
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit.ksfixtures import ldapdb
class SslTlsTest(unit.BaseTestCase):
    """Test for the SSL/TLS functionality in keystone.common.ldap.core."""

    def setUp(self):
        super(SslTlsTest, self).setUp()
        self.config_fixture = self.useFixture(config_fixture.Config(CONF))

    @mock.patch.object(common_ldap.KeystoneLDAPHandler, 'simple_bind_s')
    @mock.patch.object(ldap.ldapobject.LDAPObject, 'start_tls_s')
    def _init_ldap_connection(self, config, mock_ldap_one, mock_ldap_two):
        base_ldap = common_ldap.BaseLdap(config)
        base_ldap.get_connection()

    def test_certfile_trust_tls(self):
        handle, certfile = tempfile.mkstemp()
        self.addCleanup(os.unlink, certfile)
        self.addCleanup(os.close, handle)
        self.config_fixture.config(group='ldap', url='ldap://localhost', use_tls=True, tls_cacertfile=certfile)
        self._init_ldap_connection(CONF)
        self.assertEqual(certfile, ldap.get_option(ldap.OPT_X_TLS_CACERTFILE))

    def test_certdir_trust_tls(self):
        certdir = self.useFixture(fixtures.TempDir()).path
        self.config_fixture.config(group='ldap', url='ldap://localhost', use_tls=True, tls_cacertdir=certdir)
        self._init_ldap_connection(CONF)
        self.assertEqual(certdir, ldap.get_option(ldap.OPT_X_TLS_CACERTDIR))

    def test_certfile_trust_ldaps(self):
        handle, certfile = tempfile.mkstemp()
        self.addCleanup(os.unlink, certfile)
        self.addCleanup(os.close, handle)
        self.config_fixture.config(group='ldap', url='ldaps://localhost', use_tls=False, tls_cacertfile=certfile)
        self._init_ldap_connection(CONF)
        self.assertEqual(certfile, ldap.get_option(ldap.OPT_X_TLS_CACERTFILE))

    def test_certdir_trust_ldaps(self):
        certdir = self.useFixture(fixtures.TempDir()).path
        self.config_fixture.config(group='ldap', url='ldaps://localhost', use_tls=False, tls_cacertdir=certdir)
        self._init_ldap_connection(CONF)
        self.assertEqual(certdir, ldap.get_option(ldap.OPT_X_TLS_CACERTDIR))