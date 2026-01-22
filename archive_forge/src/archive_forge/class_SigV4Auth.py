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
class SigV4Auth(BaseSigner):
    """
    Sign a request with Signature V4.
    """
    REQUIRES_REGION = True

    def __init__(self, credentials, service_name, region_name):
        self.credentials = credentials
        self._region_name = region_name
        self._service_name = service_name

    def _sign(self, key, msg, hex=False):
        if hex:
            sig = hmac.new(key, msg.encode('utf-8'), sha256).hexdigest()
        else:
            sig = hmac.new(key, msg.encode('utf-8'), sha256).digest()
        return sig

    def headers_to_sign(self, request):
        """
        Select the headers from the request that need to be included
        in the StringToSign.
        """
        header_map = HTTPHeaders()
        for name, value in request.headers.items():
            lname = name.lower()
            if lname not in SIGNED_HEADERS_BLACKLIST:
                header_map[lname] = value
        if 'host' not in header_map:
            header_map['host'] = _host_from_url(request.url)
        return header_map

    def canonical_query_string(self, request):
        if request.params:
            return self._canonical_query_string_params(request.params)
        else:
            return self._canonical_query_string_url(urlsplit(request.url))

    def _canonical_query_string_params(self, params):
        key_val_pairs = []
        if isinstance(params, Mapping):
            params = params.items()
        for key, value in params:
            key_val_pairs.append((quote(key, safe='-_.~'), quote(str(value), safe='-_.~')))
        sorted_key_vals = []
        for key, value in sorted(key_val_pairs):
            sorted_key_vals.append(f'{key}={value}')
        canonical_query_string = '&'.join(sorted_key_vals)
        return canonical_query_string

    def _canonical_query_string_url(self, parts):
        canonical_query_string = ''
        if parts.query:
            key_val_pairs = []
            for pair in parts.query.split('&'):
                key, _, value = pair.partition('=')
                key_val_pairs.append((key, value))
            sorted_key_vals = []
            for key, value in sorted(key_val_pairs):
                sorted_key_vals.append(f'{key}={value}')
            canonical_query_string = '&'.join(sorted_key_vals)
        return canonical_query_string

    def canonical_headers(self, headers_to_sign):
        """
        Return the headers that need to be included in the StringToSign
        in their canonical form by converting all header keys to lower
        case, sorting them in alphabetical order and then joining
        them into a string, separated by newlines.
        """
        headers = []
        sorted_header_names = sorted(set(headers_to_sign))
        for key in sorted_header_names:
            value = ','.join((self._header_value(v) for v in headers_to_sign.get_all(key)))
            headers.append(f'{key}:{ensure_unicode(value)}')
        return '\n'.join(headers)

    def _header_value(self, value):
        return ' '.join(value.split())

    def signed_headers(self, headers_to_sign):
        headers = sorted((n.lower().strip() for n in set(headers_to_sign)))
        return ';'.join(headers)

    def _is_streaming_checksum_payload(self, request):
        checksum_context = request.context.get('checksum', {})
        algorithm = checksum_context.get('request_algorithm')
        return isinstance(algorithm, dict) and algorithm.get('in') == 'trailer'

    def payload(self, request):
        if self._is_streaming_checksum_payload(request):
            return STREAMING_UNSIGNED_PAYLOAD_TRAILER
        elif not self._should_sha256_sign_payload(request):
            return UNSIGNED_PAYLOAD
        request_body = request.body
        if request_body and hasattr(request_body, 'seek'):
            position = request_body.tell()
            read_chunksize = functools.partial(request_body.read, PAYLOAD_BUFFER)
            checksum = sha256()
            for chunk in iter(read_chunksize, b''):
                checksum.update(chunk)
            hex_checksum = checksum.hexdigest()
            request_body.seek(position)
            return hex_checksum
        elif request_body:
            return sha256(request_body).hexdigest()
        else:
            return EMPTY_SHA256_HASH

    def _should_sha256_sign_payload(self, request):
        if not request.url.startswith('https'):
            return True
        return request.context.get('payload_signing_enabled', True)

    def canonical_request(self, request):
        cr = [request.method.upper()]
        path = self._normalize_url_path(urlsplit(request.url).path)
        cr.append(path)
        cr.append(self.canonical_query_string(request))
        headers_to_sign = self.headers_to_sign(request)
        cr.append(self.canonical_headers(headers_to_sign) + '\n')
        cr.append(self.signed_headers(headers_to_sign))
        if 'X-Amz-Content-SHA256' in request.headers:
            body_checksum = request.headers['X-Amz-Content-SHA256']
        else:
            body_checksum = self.payload(request)
        cr.append(body_checksum)
        return '\n'.join(cr)

    def _normalize_url_path(self, path):
        normalized_path = quote(normalize_url_path(path), safe='/~')
        return normalized_path

    def scope(self, request):
        scope = [self.credentials.access_key]
        scope.append(request.context['timestamp'][0:8])
        scope.append(self._region_name)
        scope.append(self._service_name)
        scope.append('aws4_request')
        return '/'.join(scope)

    def credential_scope(self, request):
        scope = []
        scope.append(request.context['timestamp'][0:8])
        scope.append(self._region_name)
        scope.append(self._service_name)
        scope.append('aws4_request')
        return '/'.join(scope)

    def string_to_sign(self, request, canonical_request):
        """
        Return the canonical StringToSign as well as a dict
        containing the original version of all headers that
        were included in the StringToSign.
        """
        sts = ['AWS4-HMAC-SHA256']
        sts.append(request.context['timestamp'])
        sts.append(self.credential_scope(request))
        sts.append(sha256(canonical_request.encode('utf-8')).hexdigest())
        return '\n'.join(sts)

    def signature(self, string_to_sign, request):
        key = self.credentials.secret_key
        k_date = self._sign(f'AWS4{key}'.encode(), request.context['timestamp'][0:8])
        k_region = self._sign(k_date, self._region_name)
        k_service = self._sign(k_region, self._service_name)
        k_signing = self._sign(k_service, 'aws4_request')
        return self._sign(k_signing, string_to_sign, hex=True)

    def add_auth(self, request):
        if self.credentials is None:
            raise NoCredentialsError()
        datetime_now = datetime.datetime.utcnow()
        request.context['timestamp'] = datetime_now.strftime(SIGV4_TIMESTAMP)
        self._modify_request_before_signing(request)
        canonical_request = self.canonical_request(request)
        logger.debug('Calculating signature using v4 auth.')
        logger.debug('CanonicalRequest:\n%s', canonical_request)
        string_to_sign = self.string_to_sign(request, canonical_request)
        logger.debug('StringToSign:\n%s', string_to_sign)
        signature = self.signature(string_to_sign, request)
        logger.debug('Signature:\n%s', signature)
        self._inject_signature_to_request(request, signature)

    def _inject_signature_to_request(self, request, signature):
        auth_str = ['AWS4-HMAC-SHA256 Credential=%s' % self.scope(request)]
        headers_to_sign = self.headers_to_sign(request)
        auth_str.append(f'SignedHeaders={self.signed_headers(headers_to_sign)}')
        auth_str.append('Signature=%s' % signature)
        request.headers['Authorization'] = ', '.join(auth_str)
        return request

    def _modify_request_before_signing(self, request):
        if 'Authorization' in request.headers:
            del request.headers['Authorization']
        self._set_necessary_date_headers(request)
        if self.credentials.token:
            if 'X-Amz-Security-Token' in request.headers:
                del request.headers['X-Amz-Security-Token']
            request.headers['X-Amz-Security-Token'] = self.credentials.token
        if not request.context.get('payload_signing_enabled', True):
            if 'X-Amz-Content-SHA256' in request.headers:
                del request.headers['X-Amz-Content-SHA256']
            request.headers['X-Amz-Content-SHA256'] = UNSIGNED_PAYLOAD

    def _set_necessary_date_headers(self, request):
        if 'Date' in request.headers:
            del request.headers['Date']
            datetime_timestamp = datetime.datetime.strptime(request.context['timestamp'], SIGV4_TIMESTAMP)
            request.headers['Date'] = formatdate(int(calendar.timegm(datetime_timestamp.timetuple())))
            if 'X-Amz-Date' in request.headers:
                del request.headers['X-Amz-Date']
        else:
            if 'X-Amz-Date' in request.headers:
                del request.headers['X-Amz-Date']
            request.headers['X-Amz-Date'] = request.context['timestamp']