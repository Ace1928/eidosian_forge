import openstack.exceptions as exception
from openstack.identity.v3 import (
from openstack.identity.v3 import access_rule as _access_rule
from openstack.identity.v3 import credential as _credential
from openstack.identity.v3 import domain as _domain
from openstack.identity.v3 import domain_config as _domain_config
from openstack.identity.v3 import endpoint as _endpoint
from openstack.identity.v3 import federation_protocol as _federation_protocol
from openstack.identity.v3 import group as _group
from openstack.identity.v3 import identity_provider as _identity_provider
from openstack.identity.v3 import limit as _limit
from openstack.identity.v3 import mapping as _mapping
from openstack.identity.v3 import policy as _policy
from openstack.identity.v3 import project as _project
from openstack.identity.v3 import region as _region
from openstack.identity.v3 import registered_limit as _registered_limit
from openstack.identity.v3 import role as _role
from openstack.identity.v3 import role_assignment as _role_assignment
from openstack.identity.v3 import (
from openstack.identity.v3 import (
from openstack.identity.v3 import (
from openstack.identity.v3 import (
from openstack.identity.v3 import (
from openstack.identity.v3 import (
from openstack.identity.v3 import service as _service
from openstack.identity.v3 import system as _system
from openstack.identity.v3 import trust as _trust
from openstack.identity.v3 import user as _user
from openstack import proxy
from openstack import resource
from openstack import utils
def validate_user_has_domain_role(self, domain, user, role):
    """Validates that a user has a role on a domain

        :param domain: Either the ID of a domain or a
            :class:`~openstack.identity.v3.domain.Domain` instance.
        :param user: Either the ID of a user or a
            :class:`~openstack.identity.v3.user.User` instance.
        :param role: Either the ID of a role or a
            :class:`~openstack.identity.v3.role.Role` instance.
        :returns: True if user has role in domain
        """
    domain = self._get_resource(_domain.Domain, domain)
    user = self._get_resource(_user.User, user)
    role = self._get_resource(_role.Role, role)
    return domain.validate_user_has_role(self, user, role)