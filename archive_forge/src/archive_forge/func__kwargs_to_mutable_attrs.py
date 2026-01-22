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
def _kwargs_to_mutable_attrs(self, **attrs):
    update_attrs = {}
    if 'name' in attrs:
        update_attrs['name'] = attrs['name']
    if 'algorithm' in attrs:
        algorithm_value = self._algorithm_to_value(attrs['algorithm'])
        update_attrs['algorithm'] = algorithm_value
    if 'protocol' in attrs:
        update_attrs['protocol'] = self._protocol_to_value(attrs['protocol'])
    if 'port' in attrs:
        update_attrs['port'] = int(attrs['port'])
    if 'vip' in attrs:
        if attrs['vip'] == 'PUBLIC' or attrs['vip'] == 'SERVICENET':
            update_attrs['virtualIps'] = [{'type': attrs['vip']}]
        else:
            update_attrs['virtualIps'] = [{'id': attrs['vip']}]
    return update_attrs