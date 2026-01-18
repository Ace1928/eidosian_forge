from openstack.load_balancer.v2 import amphora as _amphora
from openstack.load_balancer.v2 import availability_zone as _availability_zone
from openstack.load_balancer.v2 import (
from openstack.load_balancer.v2 import flavor as _flavor
from openstack.load_balancer.v2 import flavor_profile as _flavor_profile
from openstack.load_balancer.v2 import health_monitor as _hm
from openstack.load_balancer.v2 import l7_policy as _l7policy
from openstack.load_balancer.v2 import l7_rule as _l7rule
from openstack.load_balancer.v2 import listener as _listener
from openstack.load_balancer.v2 import load_balancer as _lb
from openstack.load_balancer.v2 import member as _member
from openstack.load_balancer.v2 import pool as _pool
from openstack.load_balancer.v2 import provider as _provider
from openstack.load_balancer.v2 import quota as _quota
from openstack import proxy
from openstack import resource
def update_flavor_profile(self, flavor_profile, **attrs):
    """Update a flavor profile

        :param flavor_profile: The flavor_profile can be either the ID or a
            :class:`~openstack.load_balancer.v2.flavor_profile.FlavorProfile`
            instance
        :param dict attrs: The attributes to update on the flavor profile
            represented by ``flavor_profile``.

        :returns: The updated flavor profile
        :rtype:
            :class:`~openstack.load_balancer.v2.flavor_profile.FlavorProfile`
        """
    return self._update(_flavor_profile.FlavorProfile, flavor_profile, **attrs)