import re
import copy
import time
import base64
import warnings
from typing import List
from libcloud.pricing import get_size_price
from libcloud.utils.py3 import ET, b, basestring, ensure_string
from libcloud.utils.xml import findall, findattr, findtext, fixxpath
from libcloud.common.aws import DEFAULT_SIGNATURE_VERSION, AWSBaseResponse, SignedAWSConnection
from libcloud.common.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.iso8601 import parse_date, parse_date_allow_empty
from libcloud.utils.publickey import get_pubkey_comment, get_pubkey_ssh2_fingerprint
from libcloud.compute.providers import Provider
from libcloud.compute.constants.ec2_region_details_partial import (
class EC2ReservedNode(Node):
    """
    Class which stores information about EC2 reserved instances/nodes
    Inherits from Node and passes in None for name and private/public IPs

    Note: This class is EC2 specific.
    """

    def __init__(self, id, state, driver, size=None, image=None, extra=None):
        super().__init__(id=id, name=None, state=state, public_ips=None, private_ips=None, driver=driver, extra=extra)

    def __repr__(self):
        return '<EC2ReservedNode: id=%s>' % self.id