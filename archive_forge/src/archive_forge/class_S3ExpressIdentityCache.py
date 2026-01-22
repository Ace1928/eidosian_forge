import base64
import binascii
import datetime
import email.message
import functools
import hashlib
import io
import logging
import os
import random
import re
import socket
import time
import warnings
import weakref
from datetime import datetime as _DatetimeClass
from ipaddress import ip_address
from pathlib import Path
from urllib.request import getproxies, proxy_bypass
import dateutil.parser
from dateutil.tz import tzutc
from urllib3.exceptions import LocationParseError
import botocore
import botocore.awsrequest
import botocore.httpsession
from botocore.compat import HEX_PAT  # noqa: F401
from botocore.compat import IPV4_PAT  # noqa: F401
from botocore.compat import IPV6_ADDRZ_PAT  # noqa: F401
from botocore.compat import IPV6_PAT  # noqa: F401
from botocore.compat import LS32_PAT  # noqa: F401
from botocore.compat import UNRESERVED_PAT  # noqa: F401
from botocore.compat import ZONE_ID_PAT  # noqa: F401
from botocore.compat import (
from botocore.exceptions import (
class S3ExpressIdentityCache(IdentityCache):
    """S3Express IdentityCache for retrieving and storing
    credentials from CreateSession calls.

    This class is not intended to be instantiated in user code.
    """
    METHOD = 's3express'

    def __init__(self, client, credential_cls):
        self._client = client
        self._credential_cls = credential_cls

    @functools.lru_cache(maxsize=100)
    def get_credentials(self, bucket):
        return super().get_credentials(bucket=bucket)

    def build_refresh_callback(self, bucket):

        def refresher():
            response = self._client.create_session(Bucket=bucket)
            creds = response['Credentials']
            expiration = self._serialize_if_needed(creds['Expiration'], iso=True)
            return {'access_key': creds['AccessKeyId'], 'secret_key': creds['SecretAccessKey'], 'token': creds['SessionToken'], 'expiry_time': expiration}
        return refresher

    def _serialize_if_needed(self, value, iso=False):
        if isinstance(value, _DatetimeClass):
            if iso:
                return value.isoformat()
            return value.strftime('%Y-%m-%dT%H:%M:%S%Z')
        return value