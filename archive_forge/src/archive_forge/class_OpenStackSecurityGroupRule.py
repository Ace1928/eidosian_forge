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
class OpenStackSecurityGroupRule:
    """
    A Rule of a Security Group.
    """

    def __init__(self, id, parent_group_id, ip_protocol, from_port, to_port, driver, ip_range=None, group=None, tenant_id=None, direction=None, extra=None):
        """
        Constructor.

        :keyword    id: Rule id.
        :type       id: ``str``

        :keyword    parent_group_id: ID of the parent security group.
        :type       parent_group_id: ``str``

        :keyword    ip_protocol: IP Protocol (icmp, tcp, udp, etc).
        :type       ip_protocol: ``str``

        :keyword    from_port: Port at start of range.
        :type       from_port: ``int``

        :keyword    to_port: Port at end of range.
        :type       to_port: ``int``

        :keyword    ip_range: CIDR for address range.
        :type       ip_range: ``str``

        :keyword    group: Name of a source security group to apply to rule.
        :type       group: ``str``

        :keyword    tenant_id: Owner of the security group.
        :type       tenant_id: ``str``

        :keyword    direction: Security group Direction (ingress or egress).
        :type       direction: ``str``

        :keyword    extra: Extra attributes associated with this rule.
        :type       extra: ``dict``
        """
        self.id = id
        self.parent_group_id = parent_group_id
        self.ip_protocol = ip_protocol
        self.from_port = from_port
        self.to_port = to_port
        self.driver = driver
        self.ip_range = ''
        self.group = {}
        self.direction = 'ingress'
        if group is None:
            self.ip_range = ip_range
        else:
            self.group = {'name': group, 'tenant_id': tenant_id}
        if direction is not None:
            if direction in ['ingress', 'egress']:
                self.direction = direction
            else:
                raise OpenStackException('Security group direction incorrect value: ingress or egress.', 500, driver)
        self.tenant_id = tenant_id
        self.extra = extra or {}

    def __repr__(self):
        return '<OpenStackSecurityGroupRule id=%s parent_group_id=%s                 ip_protocol=%s from_port=%s to_port=%s>' % (self.id, self.parent_group_id, self.ip_protocol, self.from_port, self.to_port)