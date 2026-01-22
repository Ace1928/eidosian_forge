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
def _to_health_monitor(self, el):
    health_monitor_data = el['healthMonitor']
    type = health_monitor_data.get('type')
    delay = health_monitor_data.get('delay')
    timeout = health_monitor_data.get('timeout')
    attempts_before_deactivation = health_monitor_data.get('attemptsBeforeDeactivation')
    if type == 'CONNECT':
        return RackspaceHealthMonitor(type=type, delay=delay, timeout=timeout, attempts_before_deactivation=attempts_before_deactivation)
    if type == 'HTTP' or type == 'HTTPS':
        return RackspaceHTTPHealthMonitor(type=type, delay=delay, timeout=timeout, attempts_before_deactivation=attempts_before_deactivation, path=health_monitor_data.get('path'), status_regex=health_monitor_data.get('statusRegex'), body_regex=health_monitor_data.get('bodyRegex', ''))
    return None