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
class AWSRequestSignerAlgorithmV4(AWSRequestSigner):

    def get_request_params(self, params, method='GET', path='/'):
        if method == 'GET':
            params['Version'] = self.version
        return params

    def get_request_headers(self, params, headers, method='GET', path='/', data=None):
        now = datetime.utcnow()
        headers['X-AMZ-Date'] = now.strftime('%Y%m%dT%H%M%SZ')
        headers['X-AMZ-Content-SHA256'] = self._get_payload_hash(method, data)
        headers['Authorization'] = self._get_authorization_v4_header(params=params, headers=headers, dt=now, method=method, path=path, data=data)
        return (params, headers)

    def _get_authorization_v4_header(self, params, headers, dt, method='GET', path='/', data=None):
        credentials_scope = self._get_credential_scope(dt=dt)
        signed_headers = self._get_signed_headers(headers=headers)
        signature = self._get_signature(params=params, headers=headers, dt=dt, method=method, path=path, data=data)
        return 'AWS4-HMAC-SHA256 Credential=%(u)s/%(c)s, SignedHeaders=%(sh)s, Signature=%(s)s' % {'u': self.access_key, 'c': credentials_scope, 'sh': signed_headers, 's': signature}

    def _get_signature(self, params, headers, dt, method, path, data):
        key = self._get_key_to_sign_with(dt)
        string_to_sign = self._get_string_to_sign(params=params, headers=headers, dt=dt, method=method, path=path, data=data)
        return _sign(key=key, msg=string_to_sign, hex=True)

    def _get_key_to_sign_with(self, dt):
        return _sign(_sign(_sign(_sign('AWS4' + self.access_secret, dt.strftime('%Y%m%d')), self.connection.driver.region_name), self.connection.service_name), 'aws4_request')

    def _get_string_to_sign(self, params, headers, dt, method, path, data):
        canonical_request = self._get_canonical_request(params=params, headers=headers, method=method, path=path, data=data)
        return '\n'.join(['AWS4-HMAC-SHA256', dt.strftime('%Y%m%dT%H%M%SZ'), self._get_credential_scope(dt), _hash(canonical_request)])

    def _get_credential_scope(self, dt):
        return '/'.join([dt.strftime('%Y%m%d'), self.connection.driver.region_name, self.connection.service_name, 'aws4_request'])

    def _get_signed_headers(self, headers):
        return ';'.join([k.lower() for k in sorted(headers.keys(), key=str.lower)])

    def _get_canonical_headers(self, headers):
        return '\n'.join([':'.join([k.lower(), str(v).strip()]) for k, v in sorted(headers.items(), key=lambda k: k[0].lower())]) + '\n'

    def _get_payload_hash(self, method, data=None):
        if data is UnsignedPayloadSentinel:
            return UNSIGNED_PAYLOAD
        if method in ('POST', 'PUT'):
            if data:
                if hasattr(data, 'next') or hasattr(data, '__next__'):
                    return UNSIGNED_PAYLOAD
                return _hash(data)
            else:
                return UNSIGNED_PAYLOAD
        else:
            return _hash('')

    def _get_request_params(self, params):
        return '&'.join(['{}={}'.format(urlquote(k, safe=''), urlquote(str(v), safe='~')) for k, v in sorted(params.items())])

    def _get_canonical_request(self, params, headers, method, path, data):
        return '\n'.join([method, path, self._get_request_params(params), self._get_canonical_headers(headers), self._get_signed_headers(headers), self._get_payload_hash(method, data)])