from datetime import datetime
from libcloud.common.gandi import (
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def _to_iface(self, iface):
    ips = []
    for ip in iface.get('ips', []):
        new_ip = IPAddress(ip['id'], NODE_STATE_MAP.get(ip['state'], NodeState.UNKNOWN), ip['ip'], self.connection.driver, version=ip.get('version'), extra={'reverse': ip['reverse']})
        ips.append(new_ip)
    return NetworkInterface(iface['id'], NODE_STATE_MAP.get(iface['state'], NodeState.UNKNOWN), mac_address=None, driver=self.connection.driver, ips=ips, node_id=iface.get('vm_id'), extra={'bandwidth': iface['bandwidth']})