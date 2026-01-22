import ldap
from keystone.common import cache
from keystone.common import provider_api
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit.ksfixtures import ldapdb
class BaseBackendLdapIdentitySqlEverythingElseWithMapping(object):
    """Mixin base class to test mapping of default LDAP backend.

    The default configuration is not to enable mapping when using a single
    backend LDAP driver.  However, a cloud provider might want to enable
    the mapping, hence hiding the LDAP IDs from any clients of keystone.
    Setting backward_compatible_ids to False will enable this mapping.

    """

    def config_overrides(self):
        super(BaseBackendLdapIdentitySqlEverythingElseWithMapping, self).config_overrides()
        self.config_fixture.config(group='identity_mapping', backward_compatible_ids=False)