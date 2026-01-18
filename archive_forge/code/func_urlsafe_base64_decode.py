import base64
import datetime
import re
import unicodedata
from binascii import Error as BinasciiError
from email.utils import formatdate
from urllib.parse import quote, unquote
from urllib.parse import urlencode as original_urlencode
from urllib.parse import urlparse
from django.utils.datastructures import MultiValueDict
from django.utils.regex_helper import _lazy_re_compile
def urlsafe_base64_decode(s):
    """
    Decode a base64 encoded string. Add back any trailing equal signs that
    might have been stripped.
    """
    s = s.encode()
    try:
        return base64.urlsafe_b64decode(s.ljust(len(s) + len(s) % 4, b'='))
    except (LookupError, BinasciiError) as e:
        raise ValueError(e)