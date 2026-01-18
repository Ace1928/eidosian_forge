import base64
import calendar
from ipaddress import AddressValueError
from ipaddress import IPv4Address
from ipaddress import IPv6Address
import re
import struct
import time
from urllib.parse import urlparse
from saml2 import time_util
def valid_duration(val):
    try:
        time_util.parse_duration(val)
    except Exception:
        raise NotValid('duration')
    return True