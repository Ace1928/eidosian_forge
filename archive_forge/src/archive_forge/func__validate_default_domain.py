import abc
import keystone.conf
from keystone import exception
def _validate_default_domain(self, ref):
    """Validate that either the default domain or nothing is specified.

        Also removes the domain from the ref so that LDAP doesn't have to
        persist the attribute.

        """
    ref = ref.copy()
    domain_id = ref.pop('domain_id', CONF.identity.default_domain_id)
    self._validate_default_domain_id(domain_id)
    return ref