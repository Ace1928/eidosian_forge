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
class OutscaleSASNodeDriver(OutscaleNodeDriver):
    """
    Outscale SAS node driver
    """
    name = 'Outscale SAS'
    type = Provider.OUTSCALE_SAS

    def __init__(self, key, secret=None, secure=True, host=None, port=None, region='us-east-1', region_details=None, **kwargs):
        super().__init__(key=key, secret=secret, secure=secure, host=host, port=port, region=region, region_details=OUTSCALE_SAS_REGION_DETAILS, **kwargs)