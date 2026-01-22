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
def _to_subnet_association(self, element):
    """
        Parse the XML element and return a route table association object

        :rtype:     :class: `EC2SubnetAssociation`
        """
    association_id = findtext(element=element, xpath='routeTableAssociationId', namespace=NAMESPACE)
    route_table_id = findtext(element=element, xpath='routeTableId', namespace=NAMESPACE)
    subnet_id = findtext(element=element, xpath='subnetId', namespace=NAMESPACE)
    main = findtext(element=element, xpath='main', namespace=NAMESPACE)
    main = True if main else False
    return EC2SubnetAssociation(association_id, route_table_id, subnet_id, main)