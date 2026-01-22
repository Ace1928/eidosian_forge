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
def _to_connection_throttle(self, el):
    connection_throttle_data = el['connectionThrottle']
    min_connections = connection_throttle_data.get('minConnections')
    max_connections = connection_throttle_data.get('maxConnections')
    max_connection_rate = connection_throttle_data.get('maxConnectionRate')
    rate_interval = connection_throttle_data.get('rateInterval')
    return RackspaceConnectionThrottle(min_connections=min_connections, max_connections=max_connections, max_connection_rate=max_connection_rate, rate_interval_seconds=rate_interval)