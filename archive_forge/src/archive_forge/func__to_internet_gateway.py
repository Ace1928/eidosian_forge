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
def _to_internet_gateway(self, element, name=None):
    id = findtext(element=element, xpath='internetGatewayId', namespace=NAMESPACE)
    vpc_id = findtext(element=element, xpath='attachmentSet/item/vpcId', namespace=NAMESPACE)
    state = findtext(element=element, xpath='attachmentSet/item/state', namespace=NAMESPACE)
    if not state:
        state = 'available'
    tags = self._get_resource_tags(element)
    name = name if name else tags.get('Name', id)
    return VPCInternetGateway(id=id, name=name, vpc_id=vpc_id, state=state, driver=self.connection.driver, extra={'tags': tags})