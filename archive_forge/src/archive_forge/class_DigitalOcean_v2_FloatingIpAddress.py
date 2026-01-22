import json
import warnings
from libcloud.utils.py3 import httplib
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.common.digitalocean import DigitalOcean_v1_Error, DigitalOcean_v2_BaseDriver
class DigitalOcean_v2_FloatingIpAddress:
    """
    Floating IP info.
    """

    def __init__(self, id, ip_address, node_id=None, extra=None, driver=None):
        self.id = str(id)
        self.ip_address = ip_address
        self.extra = extra
        self.node_id = node_id
        self.driver = driver

    def delete(self):
        """
        Delete this floating IP

        :rtype: ``bool``
        """
        return self.driver.ex_delete_floating_ip(self)

    def __repr__(self):
        return '<DigitalOcean_v2_FloatingIpAddress: id=%s, ip_addr=%s, driver=%s>' % (self.id, self.ip_address, self.driver)