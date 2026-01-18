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
def validate_before(not_before, slack):
    if not_before:
        now = time_util.utc_now()
        nbefore = calendar.timegm(time_util.str_to_time(not_before))
        if nbefore > now + slack:
            now_str = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(now))
            raise ToEarly(f"Can't use response yet: (now={now_str} + slack={int(slack)}) <= notbefore={not_before}")
    return True