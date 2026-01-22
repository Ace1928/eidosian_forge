import sys
import hmac
import time
import uuid
import base64
import hashlib
from libcloud.utils.py3 import ET, b, u, urlquote
from libcloud.utils.xml import findtext
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
from libcloud.common.types import MalformedResponseError
class SignedAliyunConnection(AliyunConnection):
    api_version = None

    def __init__(self, user_id, key, secure=True, host=None, port=None, url=None, timeout=None, proxy_url=None, retry_delay=None, backoff=None, api_version=None, signature_version=DEFAULT_SIGNATURE_VERSION):
        super().__init__(user_id=user_id, key=key, secure=secure, host=host, port=port, url=url, timeout=timeout, proxy_url=proxy_url, retry_delay=retry_delay, backoff=backoff)
        self.signature_version = str(signature_version)
        if self.signature_version == '1.0':
            signer_cls = AliyunRequestSignerAlgorithmV1_0
        else:
            raise ValueError('Unsupported signature_version: %s' % signature_version)
        if api_version is not None:
            self.api_version = str(api_version)
        elif self.api_version is None:
            raise ValueError('Unsupported null api_version')
        self.signer = signer_cls(access_key=self.user_id, access_secret=self.key, version=self.api_version)

    def add_default_params(self, params):
        params = self.signer.get_request_params(params=params, method=self.method, path=self.action)
        return params