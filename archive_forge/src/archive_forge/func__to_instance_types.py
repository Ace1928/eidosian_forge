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
def _to_instance_types(self, elem):
    instance_types = []
    for instance_types_item in findall(element=elem, xpath='instanceTypeSet/item', namespace=OUTSCALE_NAMESPACE):
        name = findtext(element=instance_types_item, xpath='name', namespace=OUTSCALE_NAMESPACE)
        vcpu = findtext(element=instance_types_item, xpath='vcpu', namespace=OUTSCALE_NAMESPACE)
        memory = findtext(element=instance_types_item, xpath='memory', namespace=OUTSCALE_NAMESPACE)
        storageSize = findtext(element=instance_types_item, xpath='storageSize', namespace=OUTSCALE_NAMESPACE)
        storageCount = findtext(element=instance_types_item, xpath='storageCount', namespace=OUTSCALE_NAMESPACE)
        maxIpAddresses = findtext(element=instance_types_item, xpath='maxIpAddresses', namespace=OUTSCALE_NAMESPACE)
        ebsOptimizedAvailable = findtext(element=instance_types_item, xpath='ebsOptimizedAvailable', namespace=OUTSCALE_NAMESPACE)
        d = {'name': name, 'vcpu': vcpu, 'memory': memory, 'storageSize': storageSize, 'storageCount': storageCount, 'maxIpAddresses': maxIpAddresses, 'ebsOptimizedAvailable': ebsOptimizedAvailable}
        instance_types.append(d)
    return instance_types