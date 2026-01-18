import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
@property
def vpn_gateway(self):
    try:
        return self.driver.ex_list_vpn_gateways(id=self.vpn_gateway_id)[0]
    except IndexError:
        raise LibcloudError('VPN Gateway with id=%s not found' % self.vpn_gateway_id)