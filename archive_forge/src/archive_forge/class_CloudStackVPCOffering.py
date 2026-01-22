import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
class CloudStackVPCOffering:
    """
    Class representing a CloudStack VPC Offering.
    """

    def __init__(self, name, display_text, id, driver, extra=None):
        self.name = name
        self.display_text = display_text
        self.id = id
        self.driver = driver
        self.extra = extra or {}

    def __repr__(self):
        return '<CloudStackVPCOffering: name=%s, display_text=%s, id=%s, driver=%s>' % (self.name, self.display_text, self.id, self.driver.name)