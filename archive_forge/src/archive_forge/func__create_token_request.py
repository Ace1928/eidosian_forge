import base64
import binascii
import concurrent.futures
import datetime
import hashlib
import json
import math
import os
import platform
import socket
import time
import urllib.parse
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple
import urllib3
from blobfile import _common as common
from blobfile._common import (
def _create_token_request(client_email: str, private_key: str, scopes: List[str]) -> Request:
    now = time.time()
    claim_set = {'iss': client_email, 'scope': ' '.join(scopes), 'aud': 'https://www.googleapis.com/oauth2/v4/token', 'exp': now + 60 * 60, 'iat': now}
    data = {'grant_type': 'urn:ietf:params:oauth:grant-type:jwt-bearer', 'assertion': _create_jwt(private_key, claim_set)}
    return Request(url='https://www.googleapis.com/oauth2/v4/token', method='POST', headers={'Content-Type': 'application/x-www-form-urlencoded'}, data=urllib.parse.urlencode(data).encode('utf8'))