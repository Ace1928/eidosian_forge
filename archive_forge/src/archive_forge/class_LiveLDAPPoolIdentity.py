import ldappool
from keystone.common import provider_api
import keystone.conf
from keystone.identity.backends import ldap
from keystone.identity.backends.ldap import common as ldap_common
from keystone.tests import unit
from keystone.tests.unit import fakeldap
from keystone.tests.unit import test_backend_ldap_pool
from keystone.tests.unit import test_ldap_livetest
class LiveLDAPPoolIdentity(test_backend_ldap_pool.LdapPoolCommonTestMixin, test_ldap_livetest.LiveLDAPIdentity):
    """Executes existing LDAP live test with pooled LDAP handler.

    Also executes common pool specific tests via Mixin class.

    """

    def setUp(self):
        super(LiveLDAPPoolIdentity, self).setUp()
        self.addCleanup(self.cleanup_pools)
        self.conn_pools = ldap_common.PooledLDAPHandler.connection_pools

    def config_files(self):
        config_files = super(LiveLDAPPoolIdentity, self).config_files()
        config_files.append(unit.dirs.tests_conf('backend_pool_liveldap.conf'))
        return config_files

    def test_assert_connector_used_not_fake_ldap_pool(self):
        handler = ldap_common._get_connection(CONF.ldap.url, use_pool=True)
        self.assertNotEqual(type(handler.Connector), type(fakeldap.FakeLdapPool))
        self.assertEqual(type(ldappool.StateConnector), type(handler.Connector))

    def test_async_search_and_result3(self):
        self.config_fixture.config(group='ldap', page_size=1)
        self.test_user_enable_attribute_mask()

    def test_pool_size_expands_correctly(self):
        who = CONF.ldap.user
        cred = CONF.ldap.password
        ldappool_cm = self.conn_pools[CONF.ldap.url]

        def _get_conn():
            return ldappool_cm.connection(who, cred)
        with _get_conn() as c1:
            self.assertEqual(1, len(ldappool_cm))
            self.assertTrue(c1.connected)
            self.assertTrue(c1.active)
            with _get_conn() as c2:
                self.assertEqual(2, len(ldappool_cm))
                self.assertTrue(c2.connected)
                self.assertTrue(c2.active)
            self.assertEqual(2, len(ldappool_cm))
            self.assertTrue(c2.connected)
            self.assertFalse(c2.active)
            with _get_conn() as c3:
                self.assertEqual(2, len(ldappool_cm))
                self.assertTrue(c3.connected)
                self.assertTrue(c3.active)
                self.assertIs(c3, c2)
                self.assertTrue(c2.active)
                with _get_conn() as c4:
                    self.assertEqual(3, len(ldappool_cm))
                    self.assertTrue(c4.connected)
                    self.assertTrue(c4.active)

    def test_password_change_with_auth_pool_disabled(self):
        self.config_fixture.config(group='ldap', use_auth_pool=False)
        old_password = self.user_sna['password']
        self.test_password_change_with_pool()
        self.assertRaises(AssertionError, PROVIDERS.identity_api.authenticate, context={}, user_id=self.user_sna['id'], password=old_password)

    def _create_user_and_authenticate(self, password):
        user = unit.create_user(PROVIDERS.identity_api, CONF.identity.default_domain_id, password=password)
        with self.make_request():
            PROVIDERS.identity_api.authenticate(user_id=user['id'], password=password)
        return PROVIDERS.identity_api.get_user(user['id'])

    def _get_auth_conn_pool_cm(self):
        pool_url = ldap_common.PooledLDAPHandler.auth_pool_prefix + CONF.ldap.url
        return self.conn_pools[pool_url]

    def _do_password_change_for_one_user(self, password, new_password):
        self.config_fixture.config(group='ldap', use_auth_pool=True)
        self.cleanup_pools()
        self.load_backends()
        user1 = self._create_user_and_authenticate(password)
        auth_cm = self._get_auth_conn_pool_cm()
        self.assertEqual(1, len(auth_cm))
        user2 = self._create_user_and_authenticate(password)
        self.assertEqual(1, len(auth_cm))
        user3 = self._create_user_and_authenticate(password)
        self.assertEqual(1, len(auth_cm))
        user4 = self._create_user_and_authenticate(password)
        self.assertEqual(1, len(auth_cm))
        user5 = self._create_user_and_authenticate(password)
        self.assertEqual(1, len(auth_cm))
        user_api = ldap.UserApi(CONF)
        u1_dn = user_api._id_to_dn_string(user1['id'])
        u2_dn = user_api._id_to_dn_string(user2['id'])
        u3_dn = user_api._id_to_dn_string(user3['id'])
        u4_dn = user_api._id_to_dn_string(user4['id'])
        u5_dn = user_api._id_to_dn_string(user5['id'])
        auth_cm = self._get_auth_conn_pool_cm()
        with auth_cm.connection(u1_dn, password) as _:
            with auth_cm.connection(u2_dn, password) as _:
                with auth_cm.connection(u3_dn, password) as _:
                    with auth_cm.connection(u4_dn, password) as _:
                        with auth_cm.connection(u5_dn, password) as _:
                            self.assertEqual(5, len(auth_cm))
                            _.unbind_s()
        user3['password'] = new_password
        PROVIDERS.identity_api.update_user(user3['id'], user3)
        return user3

    def test_password_change_with_auth_pool_enabled_long_lifetime(self):
        self.config_fixture.config(group='ldap', auth_pool_connection_lifetime=600)
        old_password = 'my_password'
        new_password = 'new_password'
        user = self._do_password_change_for_one_user(old_password, new_password)
        user.pop('password')
        with self.make_request():
            user_ref = PROVIDERS.identity_api.authenticate(user_id=user['id'], password=old_password)
        self.assertDictEqual(user, user_ref)

    def test_password_change_with_auth_pool_enabled_no_lifetime(self):
        self.config_fixture.config(group='ldap', auth_pool_connection_lifetime=0)
        old_password = 'my_password'
        new_password = 'new_password'
        user = self._do_password_change_for_one_user(old_password, new_password)
        self.assertRaises(AssertionError, PROVIDERS.identity_api.authenticate, context={}, user_id=user['id'], password=old_password)