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
def _member_attributes(self, member):
    member_attributes = {'address': member.ip, 'port': member.port}
    member_attributes.update(self._kwargs_to_mutable_member_attrs(**member.extra))
    if 'condition' not in member_attributes:
        member_attributes['condition'] = self.CONDITION_LB_MEMBER_MAP[MemberCondition.ENABLED]
    return member_attributes