import os
import hmac
import time
import base64
import codecs
from hashlib import sha1
from libcloud.utils.py3 import ET, b, httplib, tostring, urlquote, urlencode
from libcloud.utils.xml import findtext, fixxpath
from libcloud.common.base import RawResponse, XmlResponse, ConnectionUserAndKey
from libcloud.utils.files import read_in_chunks
from libcloud.common.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.storage.base import Object, Container, StorageDriver
from libcloud.storage.types import (
@staticmethod
def _get_expires(params):
    """
        Get expires timeout seconds from parameters.
        """
    expires = None
    if 'expires' in params:
        expires = params['expires']
    elif 'Expires' in params:
        expires = params['Expires']
    if expires:
        try:
            return int(expires)
        except Exception:
            pass
    return int(time.time()) + EXPIRATION_SECONDS