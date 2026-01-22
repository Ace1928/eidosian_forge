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
class OpenStack_1_1_FloatingIpPool:
    """
    Floating IP Pool info.
    """

    def __init__(self, name, connection):
        self.name = name
        self.connection = connection

    def list_floating_ips(self):
        """
        List floating IPs in the pool

        :rtype: ``list`` of :class:`OpenStack_1_1_FloatingIpAddress`
        """
        return self._to_floating_ips(self.connection.request('/os-floating-ips').object)

    def _to_floating_ips(self, obj):
        ip_elements = obj['floating_ips']
        return [self._to_floating_ip(ip) for ip in ip_elements]

    def _to_floating_ip(self, obj):
        return OpenStack_1_1_FloatingIpAddress(id=obj['id'], ip_address=obj['ip'], pool=self, node_id=obj['instance_id'], driver=self.connection.driver)

    def get_floating_ip(self, ip):
        """
        Get specified floating IP from the pool

        :param      ip: floating IP to get
        :type       ip: ``str``

        :rtype: :class:`OpenStack_1_1_FloatingIpAddress`
        """
        for floating_ip in self.list_floating_ips():
            if floating_ip.ip_address == ip:
                return floating_ip
        return None

    def create_floating_ip(self):
        """
        Create new floating IP in the pool

        :rtype: :class:`OpenStack_1_1_FloatingIpAddress`
        """
        resp = self.connection.request('/os-floating-ips', method='POST', data={'pool': self.name})
        data = resp.object['floating_ip']
        id = data['id']
        ip_address = data['ip']
        return OpenStack_1_1_FloatingIpAddress(id=id, ip_address=ip_address, pool=self, node_id=None, driver=self.connection.driver)

    def delete_floating_ip(self, ip):
        """
        Delete specified floating IP from the pool

        :param      ip: floating IP to remove
        :type       ip: :class:`OpenStack_1_1_FloatingIpAddress`

        :rtype: ``bool``
        """
        resp = self.connection.request('/os-floating-ips/%s' % ip.id, method='DELETE')
        return resp.status in (httplib.NO_CONTENT, httplib.ACCEPTED)

    def __repr__(self):
        return '<OpenStack_1_1_FloatingIpPool: name=%s>' % self.name