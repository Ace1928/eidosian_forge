from urllib.parse import urlparse
import logging
from oauthlib.common import add_params_to_uri
from oauthlib.common import urldecode as _urldecode
from oauthlib.oauth1 import SIGNATURE_HMAC, SIGNATURE_RSA, SIGNATURE_TYPE_AUTH_HEADER
import requests
from . import OAuth1
def urldecode(body):
    """Parse query or json to python dictionary"""
    try:
        return _urldecode(body)
    except Exception:
        import json
        return json.loads(body)