import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
def transform_int_or_unlimited(value):
    try:
        return int(value)
    except ValueError as e:
        if str(value).lower() == 'unlimited':
            return -1
        raise e