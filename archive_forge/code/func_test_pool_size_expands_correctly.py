import ldappool
from keystone.common import provider_api
import keystone.conf
from keystone.identity.backends import ldap
from keystone.identity.backends.ldap import common as ldap_common
from keystone.tests import unit
from keystone.tests.unit import fakeldap
from keystone.tests.unit import test_backend_ldap_pool
from keystone.tests.unit import test_ldap_livetest
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