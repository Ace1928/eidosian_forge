import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
class CloudStackVPC:
    """
    Class representing a CloudStack VPC.
    """

    def __init__(self, name, vpc_offering_id, id, cidr, driver, zone_id=None, display_text=None, extra=None):
        self.display_text = display_text
        self.name = name
        self.vpc_offering_id = vpc_offering_id
        self.id = id
        self.zone_id = zone_id
        self.cidr = cidr
        self.driver = driver
        self.extra = extra or {}

    def __repr__(self):
        return '<CloudStackVPC: name=%s, vpc_offering_id=%s, id=%s, cidr=%s, driver=%s, zone_id=%s, display_text=%s>' % (self.name, self.vpc_offering_id, self.id, self.cidr, self.driver.name, self.zone_id, self.display_text)