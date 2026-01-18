from unittest import mock
import fixtures
import ldappool
from keystone.common import provider_api
import keystone.conf
from keystone.identity.backends import ldap
from keystone.identity.backends.ldap import common as common_ldap
from keystone.tests import unit
from keystone.tests.unit import fakeldap
from keystone.tests.unit import test_backend_ldap
@mock.patch.object(fakeldap.FakeLdap, 'search_ext')
def test_search_ext_ensure_pool_connection_released(self, mock_search_ext):
    """Test search_ext exception resiliency.

        Call search_ext function in isolation. Doing so will cause
        search_ext to borrow a connection from the pool and associate
        it with an AsynchronousMessage object. Borrowed connection ought
        to be released if anything goes wrong during LDAP API call. This
        test case intentionally throws an exception to ensure everything
        goes as expected when LDAP connection raises an exception.
        """

    class CustomDummyException(Exception):
        pass
    mock_search_ext.side_effect = CustomDummyException()
    self.config_fixture.config(group='ldap', pool_size=1)
    pool = self.conn_pools[CONF.ldap.url]
    user_api = ldap.UserApi(CONF)
    self.assertEqual(1, len(pool))
    for i in range(1, 10):
        handler = user_api.get_connection()
        self.assertIsInstance(handler.conn, common_ldap.PooledLDAPHandler)
        self.assertRaises(CustomDummyException, lambda: handler.search_ext('dc=example,dc=test', 'dummy', 'objectclass=*', ['mail', 'userPassword']))
        self.assertEqual(1, len(pool))
        with pool._pool_lock:
            for slot, conn in enumerate(pool._pool):
                self.assertFalse(conn.active)
        self.assertEqual(mock_search_ext.call_count, i)