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
def _to_interface(self, element, name=None):
    """
        Parse the XML element and return an EC2NetworkInterface object.

        :param      name: An optional name for the interface. If not provided
                          then either tag with a key "Name" or the interface ID
                          will be used (whichever is available first in that
                          order).
        :type       name: ``str``

        :rtype:     :class: `EC2NetworkInterface`
        """
    interface_id = findtext(element=element, xpath='networkInterfaceId', namespace=NAMESPACE)
    state = findtext(element=element, xpath='status', namespace=NAMESPACE)
    tags = self._get_resource_tags(element)
    name = name if name else tags.get('Name', interface_id)
    groups = self._get_security_groups(element)
    priv_ips = []
    for item in findall(element=element, xpath='privateIpAddressesSet/item', namespace=NAMESPACE):
        priv_ips.append({'private_ip': findtext(element=item, xpath='privateIpAddress', namespace=NAMESPACE), 'private_dns': findtext(element=item, xpath='privateDnsName', namespace=NAMESPACE), 'primary': findtext(element=item, xpath='primary', namespace=NAMESPACE)})
    attributes_map = RESOURCE_EXTRA_ATTRIBUTES_MAP['network_interface_attachment']
    attachment = self._get_extra_dict(element, attributes_map)
    attributes_map = RESOURCE_EXTRA_ATTRIBUTES_MAP['network_interface']
    extra = self._get_extra_dict(element, attributes_map)
    extra['tags'] = tags
    extra['attachment'] = attachment
    extra['private_ips'] = priv_ips
    extra['groups'] = groups
    return EC2NetworkInterface(interface_id, name, state, extra=extra)