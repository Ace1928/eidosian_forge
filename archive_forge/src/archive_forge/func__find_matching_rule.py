from datetime import datetime
from libcloud.utils.py3 import httplib
from libcloud.utils.misc import reverse_dict
from libcloud.common.base import JsonResponse, PollingConnection
from libcloud.common.types import LibcloudError
from libcloud.common.openstack import OpenStackDriverMixin
from libcloud.common.rackspace import AUTH_URL
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, MemberCondition
from libcloud.compute.drivers.rackspace import RackspaceConnection
def _find_matching_rule(self, rule_to_find, access_list):
    """
        LB API does not return the ID for the newly created rules, so we have
        to search the list to find the rule with a matching rule type and
        address to return an object with the right identifier.it.  The API
        enforces rule type and address uniqueness.
        """
    for r in access_list:
        if rule_to_find.rule_type == r.rule_type and rule_to_find.address == r.address:
            return r
    return None