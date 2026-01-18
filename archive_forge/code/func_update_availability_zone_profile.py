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
def update_availability_zone_profile(self, availability_zone_profile, **attrs):
    """Update an availability zone profile

        :param availability_zone_profile: The availability_zone_profile can be
            either the ID or a
            :class:`~openstack.load_balancer.v2.availability_zone_profile.AvailabilityZoneProfile`
            instance
        :param dict attrs: The attributes to update on the availability_zone
            profile represented by ``availability_zone_profile``.

        :returns: The updated availability zone profile
        :rtype:
            :class:`~openstack.load_balancer.v2.availability_zone_profile.AvailabilityZoneProfile`
        """
    return self._update(_availability_zone_profile.AvailabilityZoneProfile, availability_zone_profile, **attrs)