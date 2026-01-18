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
@mock.patch.object(fakeldap.FakeLdap, 'result3')
def test_result3_ensure_pool_connection_released(self, mock_result3):
    """Test search_ext-->result3 exception resiliency.

        Call search_ext function, grab an AsynchronousMessage object and
        call result3 with it. During the result3 call, LDAP API will throw
        an exception.The expectation is that the associated LDAP pool
        connection for AsynchronousMessage must be released back to the
        LDAP connection pool.
        """

    class CustomDummyException(Exception):
        pass
    mock_result3.side_effect = CustomDummyException()
    self.config_fixture.config(group='ldap', pool_size=1)
    pool = self.conn_pools[CONF.ldap.url]
    user_api = ldap.UserApi(CONF)
    self.assertEqual(1, len(pool))
    for i in range(1, 10):
        handler = user_api.get_connection()
        self.assertIsInstance(handler.conn, common_ldap.PooledLDAPHandler)
        msg = handler.search_ext('dc=example,dc=test', 'dummy', 'objectclass=*', ['mail', 'userPassword'])
        self.assertTrue(msg.connection.active)
        self.assertEqual(1, len(pool))
        self.assertRaises(CustomDummyException, lambda: handler.result3(msg))
        self.assertFalse(msg.connection.active)
        self.assertEqual(1, len(pool))
        self.assertEqual(mock_result3.call_count, i)