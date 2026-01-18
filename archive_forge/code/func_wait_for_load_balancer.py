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
def wait_for_load_balancer(self, name_or_id, status='ACTIVE', failures=['ERROR'], interval=2, wait=300):
    """Wait for load balancer status

        :param name_or_id: The name or ID of the load balancer.
        :param status: Desired status.
        :param failures: Statuses that would be interpreted as failures.
            Default to ['ERROR'].
        :type failures: :py:class:`list`
        :param interval: Number of seconds to wait between consecutive
            checks. Defaults to 2.
        :param wait: Maximum number of seconds to wait before the status
            to be reached. Defaults to 300.
        :returns: The load balancer is returned on success.
        :raises: :class:`~openstack.exceptions.ResourceTimeout` if transition
            to the desired status failed to occur within the specified wait
            time.
        :raises: :class:`~openstack.exceptions.ResourceFailure` if the resource
            has transited to one of the failure statuses.
        :raises: :class:`~AttributeError` if the resource does not have a
            ``status`` attribute.
        """
    lb = self._find(_lb.LoadBalancer, name_or_id, ignore_missing=False)
    return resource.wait_for_status(self, lb, status, failures, interval, wait, attribute='provisioning_status')