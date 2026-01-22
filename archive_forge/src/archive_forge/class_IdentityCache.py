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
class IdentityCache:
    """Base IdentityCache implementation for storing and retrieving
    highly accessed credentials.

    This class is not intended to be instantiated in user code.
    """
    METHOD = 'base_identity_cache'

    def __init__(self, client, credential_cls):
        self._client = client
        self._credential_cls = credential_cls

    def get_credentials(self, **kwargs):
        callback = self.build_refresh_callback(**kwargs)
        metadata = callback()
        credential_entry = self._credential_cls.create_from_metadata(metadata=metadata, refresh_using=callback, method=self.METHOD, advisory_timeout=45, mandatory_timeout=10)
        return credential_entry

    def build_refresh_callback(**kwargs):
        """Callback to be implemented by subclasses.

        Returns a set of metadata to be converted into a new
        credential instance.
        """
        raise NotImplementedError()