import base64
import binascii
import ipaddress
import json
import webbrowser
from datetime import datetime
import six
from pymacaroons import Macaroon
from pymacaroons.serializers import json_serializer
import six.moves.http_cookiejar as http_cookiejar
from six.moves.urllib.parse import urlparse
def to_bytes(s):
    """Return s as a bytes type, using utf-8 encoding if necessary.
    @param s string or bytes
    @return bytes
    """
    if isinstance(s, six.binary_type):
        return s
    if isinstance(s, six.string_types):
        return s.encode('utf-8')
    raise TypeError('want string or bytes, got {}', type(s))