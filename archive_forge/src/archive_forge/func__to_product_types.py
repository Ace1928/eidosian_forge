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
def _to_product_types(self, elem):
    product_types = []
    for product_types_item in findall(element=elem, xpath='productTypeSet/item', namespace=OUTSCALE_NAMESPACE):
        productTypeId = findtext(element=product_types_item, xpath='productTypeId', namespace=OUTSCALE_NAMESPACE)
        description = findtext(element=product_types_item, xpath='description', namespace=OUTSCALE_NAMESPACE)
        product_types.append({'productTypeId': productTypeId, 'description': description})
    return product_types