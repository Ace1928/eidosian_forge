from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.utils.misc import reverse_dict
from libcloud.common.nttcis import (
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, Provider
def _to_pool(self, element):
    pool = NttCisPool(id=element.get('id'), name=findtext(element, 'name', TYPES_URN), status=findtext(element, 'state', TYPES_URN), description=findtext(element, 'description', TYPES_URN), load_balance_method=findtext(element, 'loadBalanceMethod', TYPES_URN), health_monitor_id=findtext(element, 'healthMonitorId', TYPES_URN), service_down_action=findtext(element, 'serviceDownAction', TYPES_URN), slow_ramp_time=findtext(element, 'slowRampTime', TYPES_URN))
    return pool