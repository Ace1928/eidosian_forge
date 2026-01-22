import hmac
import time
import base64
import hashlib
from typing import Dict, Type, Optional
from hashlib import sha256
from datetime import datetime
from libcloud.utils.py3 import ET, b, httplib, urlquote, basestring, _real_unicode
from libcloud.utils.xml import findall_ignore_namespace, findtext_ignore_namespace
from libcloud.common.base import BaseDriver, XmlResponse, JsonResponse, ConnectionUserAndKey
from libcloud.common.types import InvalidCredsError, MalformedResponseError
def _get_authorization_v4_header(self, params, headers, dt, method='GET', path='/', data=None):
    credentials_scope = self._get_credential_scope(dt=dt)
    signed_headers = self._get_signed_headers(headers=headers)
    signature = self._get_signature(params=params, headers=headers, dt=dt, method=method, path=path, data=data)
    return 'AWS4-HMAC-SHA256 Credential=%(u)s/%(c)s, SignedHeaders=%(sh)s, Signature=%(s)s' % {'u': self.access_key, 'c': credentials_scope, 'sh': signed_headers, 's': signature}