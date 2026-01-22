import base64
import warnings
from libcloud.pricing import get_size_price
from libcloud.utils.py3 import ET, b, next, httplib, parse_qs, urlparse
from libcloud.utils.xml import findall
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.iso8601 import parse_date
from libcloud.common.openstack import (
from libcloud.utils.networking import is_public_subnet
from libcloud.common.exceptions import BaseHTTPError
class OpenStack_2_FloatingIpAddress(OpenStack_1_1_FloatingIpAddress):
    """
    Floating IP info 2.0.
    """

    def __init__(self, id, ip_address, pool, node_id=None, driver=None, extra=None):
        self.id = str(id)
        self.ip_address = ip_address
        self.pool = pool
        self.node_id = node_id
        self.driver = driver
        self.extra = extra if extra else {}

    def get_pool(self):
        if not self.pool:
            try:
                net = self.driver.ex_get_network(self.extra['floating_network_id'])
            except Exception:
                net = None
            if net:
                self.pool = OpenStack_2_FloatingIpPool(net.id, net.name, self.driver.network_connection)
        return self.pool

    def get_node_id(self):
        if not self.node_id:
            if 'port_details' not in self.extra or not self.extra['port_details']:
                try:
                    port = self.driver.ex_get_port(self.extra['port_id'])
                except Exception:
                    port = None
                if port:
                    self.extra['port_details'] = {'device_id': port.extra['device_id'], 'device_owner': port.extra['device_owner'], 'mac_address': port.extra['mac_address']}
            if 'port_details' in self.extra and self.extra['port_details']:
                dev_owner = self.extra['port_details']['device_owner']
                if dev_owner and dev_owner.startswith('compute:'):
                    self.node_id = self.extra['port_details']['device_id']
        return self.node_id

    def __repr__(self):
        return '<OpenStack_2_FloatingIpAddress: id=%s, ip_addr=%s, pool=%s, driver=%s>' % (self.id, self.ip_address, self.pool, self.driver)