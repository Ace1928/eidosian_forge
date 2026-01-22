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
def _to_addresses(self, response, only_associated):
    """
        Builds a list of dictionaries containing elastic IP properties.

        :param    only_associated: If true, return only those addresses
                                   that are associated with an instance.
                                   If false, return all addresses.
        :type     only_associated: ``bool``

        :rtype:   ``list`` of :class:`ElasticIP`
        """
    addresses = []
    for el in response.findall(fixxpath(xpath='addressesSet/item', namespace=NAMESPACE)):
        addr = self._to_address(el, only_associated)
        if addr is not None:
            addresses.append(addr)
    return addresses