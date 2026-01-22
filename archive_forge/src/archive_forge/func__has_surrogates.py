import os
import re
import time
import random
import socket
import datetime
import urllib.parse
from email._parseaddr import quote
from email._parseaddr import AddressList as _AddressList
from email._parseaddr import mktime_tz
from email._parseaddr import parsedate, parsedate_tz, _parsedate_tz
from email.charset import Charset
def _has_surrogates(s):
    """Return True if s may contain surrogate-escaped binary data."""
    try:
        s.encode()
        return False
    except UnicodeEncodeError:
        return True