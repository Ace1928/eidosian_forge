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
class SignedAWSConnection(AWSTokenConnection):
    version = None

    def __init__(self, user_id, key, secure=True, host=None, port=None, url=None, timeout=None, proxy_url=None, token=None, retry_delay=None, backoff=None, signature_version=DEFAULT_SIGNATURE_VERSION):
        super().__init__(user_id=user_id, key=key, secure=secure, host=host, port=port, url=url, timeout=timeout, token=token, retry_delay=retry_delay, backoff=backoff, proxy_url=proxy_url)
        self.signature_version = str(signature_version)
        if self.signature_version == '2':
            signer_cls = AWSRequestSignerAlgorithmV2
        elif self.signature_version == '4':
            signer_cls = AWSRequestSignerAlgorithmV4
        else:
            raise ValueError('Unsupported signature_version: %s' % signature_version)
        self.signer = signer_cls(access_key=self.user_id, access_secret=self.key, version=self.version, connection=self)

    def add_default_params(self, params):
        params = self.signer.get_request_params(params=params, method=self.method, path=self.action)
        for key, value in params.items():
            if not isinstance(value, (_real_unicode, basestring, int, bool)):
                msg = PARAMS_NOT_STRING_ERROR_MSG % (key, value, type(value))
                raise ValueError(msg)
        return params

    def pre_connect_hook(self, params, headers):
        params, headers = self.signer.get_request_headers(params=params, headers=headers, method=self.method, path=self.action, data=self.data)
        return (params, headers)