import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
class CloudStackVpnGateway:
    """
    Class representing a CloudStack VPN Gateway.
    """

    def __init__(self, id, account, domain, domain_id, public_ip, vpc_id, driver, extra=None):
        self.id = id
        self.account = account
        self.domain = domain
        self.domain_id = domain_id
        self.public_ip = public_ip
        self.vpc_id = vpc_id
        self.driver = driver
        self.extra = extra or {}

    @property
    def vpc(self):
        for vpc in self.driver.ex_list_vpcs():
            if self.vpc_id == vpc.id:
                return vpc
        raise LibcloudError('VPC with id=%s not found' % self.vpc_id)

    def delete(self):
        return self.driver.ex_delete_vpn_gateway(vpn_gateway=self)

    def __repr__(self):
        return '<CloudStackVpnGateway: account=%s, domain=%s, domain_id=%s, id=%s, public_ip=%s, vpc_id=%s, driver=%s>' % (self.account, self.domain, self.domain_id, self.id, self.public_ip, self.vpc_id, self.driver.name)