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
def _to_device_mapping(self, element):
    """
        Parse the XML element and return a dictionary of device properties.
        Additional information can be found at http://goo.gl/GjWYBf.

        @note: EBS volumes do not have a virtual name. Only ephemeral
               disks use this property.
        :rtype:     ``dict``
        """
    mapping = {}
    mapping['device_name'] = findattr(element=element, xpath='deviceName', namespace=NAMESPACE)
    mapping['virtual_name'] = findattr(element=element, xpath='virtualName', namespace=NAMESPACE)
    if mapping['virtual_name'] is None:
        mapping['ebs'] = self._get_extra_dict(element, RESOURCE_EXTRA_ATTRIBUTES_MAP['ebs_volume'])
    return mapping