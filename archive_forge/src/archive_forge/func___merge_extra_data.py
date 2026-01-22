import hmac
import json
import base64
import datetime
from hashlib import sha256
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import (
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, ProviderError, InvalidCredsError
def __merge_extra_data(self, rdata, extra):
    if extra is not None:
        for param in VALID_RECORD_PARAMS_EXTRA:
            if param in extra:
                rdata[param] = extra[param]
    return rdata