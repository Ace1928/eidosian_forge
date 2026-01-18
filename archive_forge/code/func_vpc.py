import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
@property
def vpc(self):
    for vpc in self.driver.ex_list_vpcs():
        if self.vpc_id == vpc.id:
            return vpc
    raise LibcloudError('VPC with id=%s not found' % self.vpc_id)