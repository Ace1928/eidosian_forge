import base64
import calendar
import datetime
import functools
import hmac
import json
import logging
import time
from collections.abc import Mapping
from email.utils import formatdate
from hashlib import sha1, sha256
from operator import itemgetter
from botocore.compat import (
from botocore.exceptions import NoAuthTokenError, NoCredentialsError
from botocore.utils import (
from botocore.compat import MD5_AVAILABLE  # noqa
class HmacV1Auth(BaseSigner):
    QSAOfInterest = ['accelerate', 'acl', 'cors', 'defaultObjectAcl', 'location', 'logging', 'partNumber', 'policy', 'requestPayment', 'torrent', 'versioning', 'versionId', 'versions', 'website', 'uploads', 'uploadId', 'response-content-type', 'response-content-language', 'response-expires', 'response-cache-control', 'response-content-disposition', 'response-content-encoding', 'delete', 'lifecycle', 'tagging', 'restore', 'storageClass', 'notification', 'replication', 'requestPayment', 'analytics', 'metrics', 'inventory', 'select', 'select-type', 'object-lock']

    def __init__(self, credentials, service_name=None, region_name=None):
        self.credentials = credentials

    def sign_string(self, string_to_sign):
        new_hmac = hmac.new(self.credentials.secret_key.encode('utf-8'), digestmod=sha1)
        new_hmac.update(string_to_sign.encode('utf-8'))
        return encodebytes(new_hmac.digest()).strip().decode('utf-8')

    def canonical_standard_headers(self, headers):
        interesting_headers = ['content-md5', 'content-type', 'date']
        hoi = []
        if 'Date' in headers:
            del headers['Date']
        headers['Date'] = self._get_date()
        for ih in interesting_headers:
            found = False
            for key in headers:
                lk = key.lower()
                if headers[key] is not None and lk == ih:
                    hoi.append(headers[key].strip())
                    found = True
            if not found:
                hoi.append('')
        return '\n'.join(hoi)

    def canonical_custom_headers(self, headers):
        hoi = []
        custom_headers = {}
        for key in headers:
            lk = key.lower()
            if headers[key] is not None:
                if lk.startswith('x-amz-'):
                    custom_headers[lk] = ','.join((v.strip() for v in headers.get_all(key)))
        sorted_header_keys = sorted(custom_headers.keys())
        for key in sorted_header_keys:
            hoi.append(f'{key}:{custom_headers[key]}')
        return '\n'.join(hoi)

    def unquote_v(self, nv):
        """
        TODO: Do we need this?
        """
        if len(nv) == 1:
            return nv
        else:
            return (nv[0], unquote(nv[1]))

    def canonical_resource(self, split, auth_path=None):
        if auth_path is not None:
            buf = auth_path
        else:
            buf = split.path
        if split.query:
            qsa = split.query.split('&')
            qsa = [a.split('=', 1) for a in qsa]
            qsa = [self.unquote_v(a) for a in qsa if a[0] in self.QSAOfInterest]
            if len(qsa) > 0:
                qsa.sort(key=itemgetter(0))
                qsa = ['='.join(a) for a in qsa]
                buf += '?'
                buf += '&'.join(qsa)
        return buf

    def canonical_string(self, method, split, headers, expires=None, auth_path=None):
        cs = method.upper() + '\n'
        cs += self.canonical_standard_headers(headers) + '\n'
        custom_headers = self.canonical_custom_headers(headers)
        if custom_headers:
            cs += custom_headers + '\n'
        cs += self.canonical_resource(split, auth_path=auth_path)
        return cs

    def get_signature(self, method, split, headers, expires=None, auth_path=None):
        if self.credentials.token:
            del headers['x-amz-security-token']
            headers['x-amz-security-token'] = self.credentials.token
        string_to_sign = self.canonical_string(method, split, headers, auth_path=auth_path)
        logger.debug('StringToSign:\n%s', string_to_sign)
        return self.sign_string(string_to_sign)

    def add_auth(self, request):
        if self.credentials is None:
            raise NoCredentialsError
        logger.debug('Calculating signature using hmacv1 auth.')
        split = urlsplit(request.url)
        logger.debug('HTTP request method: %s', request.method)
        signature = self.get_signature(request.method, split, request.headers, auth_path=request.auth_path)
        self._inject_signature(request, signature)

    def _get_date(self):
        return formatdate(usegmt=True)

    def _inject_signature(self, request, signature):
        if 'Authorization' in request.headers:
            del request.headers['Authorization']
        auth_header = f'AWS {self.credentials.access_key}:{signature}'
        request.headers['Authorization'] = auth_header