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
def valid_unsigned_short(val):
    try:
        struct.pack('H', int(val))
    except struct.error:
        raise NotValid('unsigned short')
    except ValueError:
        raise NotValid('unsigned short')
    return True