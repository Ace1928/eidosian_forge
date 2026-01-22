import collections
import copy
import datetime
import json
import logging
import os
import shutil
import socket
import sys
import tempfile
import six
from six.moves import http_client
from six.moves import urllib
import oauth2client
from oauth2client import _helpers
from oauth2client import _pkce
from oauth2client import clientsecrets
from oauth2client import transport
def _parse_exchange_token_response(content):
    """Parses response of an exchange token request.

    Most providers return JSON but some (e.g. Facebook) return a
    url-encoded string.

    Args:
        content: The body of a response

    Returns:
        Content as a dictionary object. Note that the dict could be empty,
        i.e. {}. That basically indicates a failure.
    """
    resp = {}
    content = _helpers._from_bytes(content)
    try:
        resp = json.loads(content)
    except Exception:
        resp = _helpers.parse_unique_urlencoded(content)
    if resp and 'expires' in resp:
        resp['expires_in'] = resp.pop('expires')
    return resp