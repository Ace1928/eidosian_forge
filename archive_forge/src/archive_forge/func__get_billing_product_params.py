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
def _get_billing_product_params(self, billing_products):
    """
        Return a list of dictionaries with valid param for billing product.

        :param      billing_product: List of billing code values(str)
        :type       billing product: ``list``

        :return:    Dictionary representation of the billing product codes
        :rtype:     ``dict``
        """
    if not isinstance(billing_products, (list, tuple)):
        raise AttributeError('billing_products not list or tuple')
    params = {}
    for idx, v in enumerate(billing_products):
        idx += 1
        params['BillingProduct.%d' % idx] = str(v)
    return params