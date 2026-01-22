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
def _epoch_seconds_to_datetime(value, tzinfo):
    """Parse numerical epoch timestamps (seconds since 1970) into a
    ``datetime.datetime`` in UTC using ``datetime.timedelta``. This is intended
    as fallback when ``fromtimestamp`` raises ``OverflowError`` or ``OSError``.

    :type value: float or int
    :param value: The Unix timestamps as number.

    :type tzinfo: callable
    :param tzinfo: A ``datetime.tzinfo`` class or compatible callable.
    """
    epoch_zero = datetime.datetime(1970, 1, 1, 0, 0, 0, tzinfo=tzutc())
    epoch_zero_localized = epoch_zero.astimezone(tzinfo())
    return epoch_zero_localized + datetime.timedelta(seconds=value)