import ldap
from keystone.common import cache
from keystone.common import provider_api
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit.ksfixtures import ldapdb
class BaseBackendLdap(object):
    """Mixin class to set up an all-LDAP configuration."""

    def setUp(self):
        self.useFixture(database.Database())
        super(BaseBackendLdap, self).setUp()

    def load_fixtures(self, fixtures):
        create_group_container(PROVIDERS.identity_api)
        super(BaseBackendLdap, self).load_fixtures(fixtures)