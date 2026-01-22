import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
class CloudStackVpnConnection:
    """
    Class representing a CloudStack VPN Connection.
    """

    def __init__(self, id, passive, vpn_customer_gateway_id, vpn_gateway_id, state, driver, extra=None):
        self.id = id
        self.passive = passive
        self.vpn_customer_gateway_id = vpn_customer_gateway_id
        self.vpn_gateway_id = vpn_gateway_id
        self.state = state
        self.driver = driver
        self.extra = extra or {}

    @property
    def vpn_customer_gateway(self):
        try:
            return self.driver.ex_list_vpn_customer_gateways(id=self.vpn_customer_gateway_id)[0]
        except IndexError:
            raise LibcloudError('VPN Customer Gateway with id=%s not found' % self.vpn_customer_gateway_id)

    @property
    def vpn_gateway(self):
        try:
            return self.driver.ex_list_vpn_gateways(id=self.vpn_gateway_id)[0]
        except IndexError:
            raise LibcloudError('VPN Gateway with id=%s not found' % self.vpn_gateway_id)

    def delete(self):
        return self.driver.ex_delete_vpn_connection(vpn_connection=self)

    def __repr__(self):
        return '<CloudStackVpnConnection: id=%s, passive=%s, vpn_customer_gateway_id=%s, vpn_gateway_id=%s, state=%s, driver=%s>' % (self.id, self.passive, self.vpn_customer_gateway_id, self.vpn_gateway_id, self.state, self.driver.name)