import ldappool
from keystone.common import provider_api
import keystone.conf
from keystone.identity.backends import ldap
from keystone.identity.backends.ldap import common as ldap_common
from keystone.tests import unit
from keystone.tests.unit import fakeldap
from keystone.tests.unit import test_backend_ldap_pool
from keystone.tests.unit import test_ldap_livetest
def _get_auth_conn_pool_cm(self):
    pool_url = ldap_common.PooledLDAPHandler.auth_pool_prefix + CONF.ldap.url
    return self.conn_pools[pool_url]