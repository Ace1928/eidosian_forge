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
def _url_has_allowed_host_and_scheme(url, allowed_hosts, require_https=False):
    if url.startswith('///'):
        return False
    try:
        url_info = urlparse(url)
    except ValueError:
        return False
    if not url_info.netloc and url_info.scheme:
        return False
    if unicodedata.category(url[0])[0] == 'C':
        return False
    scheme = url_info.scheme
    if not url_info.scheme and url_info.netloc:
        scheme = 'http'
    valid_schemes = ['https'] if require_https else ['http', 'https']
    return (not url_info.netloc or url_info.netloc in allowed_hosts) and (not scheme or scheme in valid_schemes)