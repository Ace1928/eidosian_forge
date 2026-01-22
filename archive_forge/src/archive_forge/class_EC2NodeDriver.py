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
class EC2NodeDriver(BaseEC2NodeDriver):
    """
    Amazon EC2 node driver.
    """
    connectionCls = EC2Connection
    type = Provider.EC2
    name = 'Amazon EC2'
    website = 'http://aws.amazon.com/ec2/'
    path = '/'
    NODE_STATE_MAP = {'pending': NodeState.PENDING, 'running': NodeState.RUNNING, 'shutting-down': NodeState.UNKNOWN, 'terminated': NodeState.TERMINATED, 'stopped': NodeState.STOPPED}

    def __init__(self, key, secret=None, secure=True, host=None, port=None, region='us-east-1', token=None, signature_version=None, **kwargs):
        if hasattr(self, '_region'):
            region = self._region
        valid_regions = self.list_regions()
        if region not in valid_regions:
            raise ValueError('Invalid region: %s' % region)
        details = REGION_DETAILS_PARTIAL[region]
        self.region_name = region
        self.token = token
        self.api_name = details['api_name']
        self.country = details['country']
        if signature_version:
            self.signature_version = signature_version
        else:
            self.signature_version = details.get('signature_version', DEFAULT_SIGNATURE_VERSION)
        host = host or details['endpoint']
        super().__init__(key=key, secret=secret, secure=secure, host=host, port=port, **kwargs)

    @classmethod
    def list_regions(cls):
        return VALID_EC2_REGIONS