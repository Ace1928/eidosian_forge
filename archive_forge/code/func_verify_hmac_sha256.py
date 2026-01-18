import binascii
import hashlib
import hmac
import ipaddress
import logging
import urllib.parse as urlparse
import warnings
from oauthlib.common import extract_params, safe_string_equals, urldecode
from . import utils
def verify_hmac_sha256(request, client_secret=None, resource_owner_secret=None):
    return _verify_hmac('SHA-256', request, client_secret, resource_owner_secret)