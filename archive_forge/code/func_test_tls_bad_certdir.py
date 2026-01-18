import ldap.modlist
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import identity
from keystone.tests import unit
from keystone.tests.unit import test_ldap_livetest
def test_tls_bad_certdir(self):
    self.config_fixture.config(group='ldap', use_tls=True, tls_cacertfile=None, tls_req_cert='demand', tls_cacertdir='/etc/keystone/ssl/mythicalcertdir')
    PROVIDERS.identity_api = identity.backends.ldap.Identity()
    user = unit.new_user_ref('default')
    self.assertRaises(IOError, PROVIDERS.identity_api.create_user, user)